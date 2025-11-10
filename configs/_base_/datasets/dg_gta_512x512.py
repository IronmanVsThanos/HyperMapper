_base_ = [
    # "./gta_742x742.py",
    # "./gta_512x512.py",
    "./gta_784x784.py",
    # # "./bdd100k_512x512.py",
    # "./cityscapes_512x512.py",
    # "./cityscapes_742x742.py",
    "./cityscapes_784x784.py",
    # "./tsp6k_512x512.py",
    # "./tsp6k_742x742.py",
    "./tsp6k_784x784.py",
    # "./tsp6k_848x848.py",
    # "./tsp6k_880x880.py",
    # "./tsp6k_1024x1024.py",
    # "./team_512x512.py",
    # "./snow-acdc_1024x1024.py",
    # "./rain-acdc_1024x1024.py",
    # "./fog-acdc_1024x1024.py",
    # "./night-acdc_1024x1024.py",
    # "./tsp6k_1036x1036.py",
    # "./urbansyn_512x512.py",
    # "./1_16citys_512x512.py",
    # "./1_16tsp6k_512x512.py",
    # "./mapillary_512x512.py",
    # "./tsp6k_supervised_cityscapes_512x512.py"
]
# #os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    # dataset={{_base_.train_gta}},
    dataset=dict(
            type="ConcatDataset",
            datasets=[
                {{_base_.train_gta}},
                # {{_base_.train_cityscapes_1_16}},
                # {{_base_.train_tsp_1_16}},
                # {{_base_.train_cityscapes}},
                # {{_base_.train_urbansyn}},
                # {{_base_.train_team_seg}},

                # {{_base_.train_rain_acdc}},
                # {{_base_.train_fog_acdc}},
                # {{_base_.train_night_acdc}},
                # {{_base_.train_rain_acdc}},
                # {{_base_.train_tsp6k}},
                # {{_base_.val_bdd}},
                # {{_base_.val_mapillary}},
            ],
        ),
)


val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_cityscapes}},
            {{_base_.val_tsp6k}},
            # {{_base_.val_rain_acdc}},
            # {{_base_.val_fog_acdc}},
            # {{_base_.val_night_acdc}},
            # {{_base_.val_rain_acdc}},
            # {{_base_.val_team_seg}}
            # {{_base_.val_bdd}},
            # {{_base_.val_mapillary}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["city", "tsp", "fog_acdc", "night_acdc", "rain_acdc"]
    # type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "tsp", "team"]
# type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["team_seg"]
)
test_evaluator=val_evaluator
