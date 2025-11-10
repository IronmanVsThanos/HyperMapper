crop_size = (518, 518)
num_classes = 19
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
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
        output_mode='seg',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data/DL/code/Rein-train-depth/checkpoints/depth_anything_v2_vitl_with_meta_modified_renamed.pth'
        )
    ),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[1024, 1024, 1024],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        crop_size=(512, 512),
        stride=(341, 341),
    ),
)
