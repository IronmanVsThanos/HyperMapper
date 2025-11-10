tsp_cityscapes_type = "CityscapesDataset"
tsp_cityscapes_root = "/data/DL/code/Rein/data/tsp_back/"
tsp_cityscapes_crop_size = (518, 518)
tsp6k_cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1036, 518)),
    dict(type="RandomCrop", crop_size=tsp_cityscapes_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
tsp6k_cityscapes_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1036, 518), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type="RandomCrop", crop_size=cityscapes_crop_size, cat_max_ratio=0.75),#11
    # dict(type="RandomFlip", prob=0.5),#11
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_tsp6k_cityscapes = dict(
    type=tsp_cityscapes_type,
    data_root=tsp_cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/train",
        seg_map_path="gtFine/train_ori",
    ),
    img_suffix=".jpg",
    seg_map_suffix="_sem.png",
    pipeline=tsp6k_cityscapes_train_pipeline,
)
train_dataloader=dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_tsp6k_cityscapes,
)
val_tsp6k_cityscapes = dict(
    type=tsp_cityscapes_type,
    data_root=tsp_cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/val/",
        seg_map_path="gtFine/val_ori/",
    ),
    img_suffix=".jpg",
    seg_map_suffix="_sem.png",
    pipeline=tsp6k_cityscapes_test_pipeline,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_tsp6k_cityscapes,
)
val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU"],
)
test_dataloader=val_dataloader
test_evaluator=val_evaluator