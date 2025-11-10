import torch
import torch.nn as nn

from .dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention
from .dinov2_layers.block import NestedTensorBlock as Block
from .dpt_dinov2 import DINOv2
from mmengine.registry import MODELS
# from mmseg.models.builder import BACKBONES
from mmengine.model import BaseModule
# from mmseg.models.builder import MODELS
from mmseg.registry import MODELS

@MODELS.register_module()
class DepthAnythingBackbone(BaseModule):
    def __init__(self,
                 encoder='vitl',
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.out_channels = out_channels

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            checkpoint = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            # 加载pretrained部分的权重
            backbone_state_dict = {}
            for k, v in checkpoint.items():
                if 'pretrained' in k:
                    # 去掉pretrained前缀
                    new_key = k.replace('pretrained.', '')
                    backbone_state_dict[new_key] = v
            self.pretrained.load_state_dict(backbone_state_dict, strict=False)
    def forward(self, x):
        ret = self.pretrained.forward_features(x)
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder],
                                                           return_class_token=True)
        return features, (patch_h, patch_w)