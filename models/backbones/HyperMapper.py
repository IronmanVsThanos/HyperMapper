import math
from functools import reduce
from operator import mul
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from mmseg.models.builder import MODELS

@MODELS.register_module()
class HyperMapper(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        latent_dim: int = 256,  # 中等压缩 latent_dim
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        learnable_c: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.latent_dim = latent_dim
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.learnable_c = learnable_c
        self.create_model()

    def create_model(self):
        if self.learnable_c:
            self.manifold = geoopt.PoincareBall(c=nn.Parameter(torch.tensor(0.01)))
        else:
            self.manifold = geoopt.PoincareBall(c=0.01)

        # 小latent token
        self.learnable_tokens_latent = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.latent_dim])
        )
        self.token_proj = nn.Linear(self.latent_dim, self.embed_dims)

        self.scale = nn.Parameter(torch.tensor(self.scale_init))

        # 更小的MLP bottleneck (1/4 hidden size)
        hidden_dim = self.embed_dims // 4
        self.mlp_token2feat = nn.Sequential(
            nn.Linear(self.embed_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embed_dims)
        )
        self.mlp_delta_f = nn.Sequential(
            nn.Linear(self.embed_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embed_dims)
        )

        val = math.sqrt(
            6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.latent_dim)
        )
        nn.init.uniform_(self.learnable_tokens_latent.data, -val, val)

        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)

        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            for m in self.mlp_delta_f.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            tokens = self.learnable_tokens_latent
        else:
            tokens = self.learnable_tokens_latent[layer]
        tokens = self.token_proj(tokens)  # latent → embed_dims
        return tokens

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        tokens = self.get_tokens(layer)

        feats_hyper = self.manifold.expmap0(feats)
        tokens_hyper = self.manifold.expmap0(tokens)

        delta_feat = self.forward_delta_feat(feats_hyper, tokens_hyper, layer)
        delta_feat = delta_feat * self.scale

        feats_hyper = self.manifold.mobius_add(feats_hyper, delta_feat)

        feats = self.manifold.logmap0(feats_hyper)

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )

        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f


@MODELS.register_module()
class LoRAHyperMapper(HyperMapper):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

        del self.learnable_tokens_latent

        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.latent_dim])
        )
        self.token_proj = nn.Linear(self.latent_dim, self.embed_dims)

        val = math.sqrt(
            6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + (self.lora_dim * self.latent_dim) ** 0.5)
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            tokens_latent = self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            tokens_latent = self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
        tokens = self.token_proj(tokens_latent)
        return tokens





