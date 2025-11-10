# # configs/_base_/models/depth_anything_v2.py
#
# # model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=dict(
#         type='SegDataPreProcessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True,
#         pad_val=0,
#         seg_pad_val=255,
#         size=(518, 518)),  # 确保输入尺寸正确
#     backbone=dict(
#         type='DepthAnythingBackbone',
#         encoder='vitl',
#         features=256,
#         out_channels=[256, 512, 1024, 1024],
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='G:/Code/workspace/Rein-train/checkpoints/depth_anything_v2_vitl.pth'
#         )
#     ),
#     decode_head=dict(
#         type='DepthAnythingHead',
#         in_channels=1024,
#         features=256,
#         out_channels=[256, 512, 1024, 1024],
#         align_corners=False,  # 添加这一行
#         num_classes=1,  # 添加这一行
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='G:/Code/workspace/Rein-train/checkpoints/depth_anything_v2_vitl.pth'
#         )
#     ),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))
#
# # 添加其他必要的配置
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(518, 518)),
#     dict(type='PackSegInputs')
# ]
#
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='BaseSegDataset',
#         pipeline=test_pipeline))
#
# test_evaluator = dict(type='BaseMetric')
# test_cfg = dict(type='TestLoop')
# default_scope = 'mmseg'
# configs/_base_/models/depth_anything_v2.py

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(518, 518)),
    backbone=dict(
        type='DPTDinoVisionTransformer',
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,  # for ViT-L
        depth=24,        # for ViT-L
        num_heads=16,    # for ViT-L
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        init_values=1e-5,
        num_register_tokens=0,
        out_indices=[7, 11, 15, 23],
        output_mode='depth',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='G:/Code/workspace/Rein-train/checkpoints/depth_anything_v2_vitl.pth'
        )
    ),
    decode_head=dict(
        type='DepthAnythingHead',
        in_channels=1024,
        features=256,
        out_channels=[256, 512, 1024, 1024],
        align_corners=False,
        num_classes=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='G:/Code/workspace/Rein-train/checkpoints/depth_anything_v2_vitl.pth'
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# 添加其他必要的配置
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(518, 518)),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        pipeline=test_pipeline))

test_evaluator = dict(type='BaseMetric')
test_cfg = dict(type='TestLoop')
default_scope = 'mmseg'