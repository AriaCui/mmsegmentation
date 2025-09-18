_base_ = [
    'mmsegmentation-main/configs/_base_/default_runtime.py'
]
work_dir = 'workdir/deeplabv3plus'

custom_imports = dict(
    imports=['mmseg.datasets.medical_dataset', 'mmseg.datasets.transforms.loading', 'mmseg.evaluation.metrics.mia_seg_metric'],
    allow_failed_imports=False
)

crop_size = (960, 544)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size)

pretrained = "pretrained/backbone/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth"
num_classes = 2

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained,
                      prefix='backbone.'),
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 12, 24, 36),
        c1_in_channels=64,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

reduce_zero_label = False
train_pipeline_label = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsModified', reduce_zero_label=reduce_zero_label, num_classes=num_classes),
    dict(
        type='RandomResize',
        scale=(960, 544),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='BioMedicalGaussianBlur', sigma_range=(0.1, 2.0), prob=0.5),
    # dict(type='RandomCutOut', prob=0.5, n_holes=1, cutout_ratio=(0.1, 0.3)),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 544), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotationsModified', reduce_zero_label=reduce_zero_label, num_classes=num_classes),
    dict(type='PackSegInputs')
]

# dataset settings
dataset_type = 'MedicalSegDataset'
data_root_label = ""

METAINFO = dict(
    classes=('bg', 'rlnever'),
    palette=[[0, 0, 0], [0, 0, 255]],
    label_map={0: 0, 1: 1}
)

train_dataset_labeled = dict(
    type=dataset_type,
    metainfo=METAINFO,
    ann_file="data/train_images/train_resize_data.txt",
    reduce_zero_label=reduce_zero_label,
    pipeline=train_pipeline_label)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=train_dataset_labeled
)

test_dataset_labeled = dict(
    type=dataset_type,
    metainfo=METAINFO,
    ann_file="data/test_images/test_resize_data.txt",
    reduce_zero_label=reduce_zero_label,
    pipeline=test_pipeline)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset_labeled)

test_dataloader = val_dataloader
val_evaluator = dict(type='MIASegMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

max_epochs = 100
val_interval = 1

optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))

optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning policy
param_scheduler = [
    dict(
        begin=0,
        end=max_epochs,
        by_epoch=True,
        eta_min=1e-6,
        type='CosineAnnealingLR'),
]

train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=val_interval, max_keep_ckpts=1, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

resume = False
load_from = None
find_unused_parameters = True
