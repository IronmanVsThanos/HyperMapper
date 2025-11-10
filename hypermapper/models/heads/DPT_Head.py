import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule  # 改用 mmengine 中的 BaseModule
from mmseg.models.builder import MODELS  # 新的注册器导入方式
from ..util.blocks import FeatureFusionBlock, _make_scratch


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


@MODELS.register_module()
class DepthAnythingHead(BaseModule):
    def __init__(self,
                 in_channels,
                 features=256,
                 use_bn=False,
                 out_channels=[256, 512, 1024, 1024],
                 use_clstoken=False,
                 align_corners=False,
                 num_classes=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.align_corners = align_corners
        self.num_classes = num_classes
        self.use_clstoken = use_clstoken
        self.channels = out_channels  # 保存中间层的通道数
        self.out_channels = 1  # 最终输出通道数为1

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channel,  # 使用 self.channels
                kernel_size=1,
                stride=1,
                padding=0,
            ) for channel in self.channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=self.channels[1],
                out_channels=self.channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=self.channels[3],
                out_channels=self.channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            self.channels,  # 使用 self.channels
            features,
            groups=1,
            expand=False,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, self.out_channels, kernel_size=1, stride=1, padding=0),  # 输出通道为1
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, inputs):
        features, patch_hw = inputs
        patch_h, patch_w = patch_hw
        out = []
        for i, x in enumerate(features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)),
                            mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        depth = F.relu(out)

        return dict(pred_depth=depth.squeeze(1))
    def predict(self, features, batch_img_metas, test_cfg):
        """Inference with full image."""
        depth_pred = self.forward(features)
        return depth_pred

    def loss(self, inputs, batch_data_samples, train_cfg):
        """Forward function for training.
        在训练时使用，如果只做推理可以返回空的损失
        """
        depth_pred = self.forward(inputs)
        losses = dict()
        return losses