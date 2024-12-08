ann_file_train = '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/trainPrivacyRaw.txt'
ann_file_val = '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/valPrivacyRaw.txt'
auto_scale_lr = dict(base_batch_size=256, enable=False)
data_root = '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/data'
data_root_val = '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/data'
dataset_type = 'RawframeDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = '/home/node1/Desktop/code/ai/open-mmlab/mmaction2/work_dirs/resnet_cls_softmax_privacy_08-zqf/epoch_150.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=False,
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        type='ResNet'),
    cls_head=dict(
        average_clips='score',
        in_channels=2048,
        init_std=0.01,
        num_classes=13,
        type='SoftmaxHead'),
    data_preprocessor=dict(
        format_shape='NCHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=None,
    train_cfg=None,
    type='Recognizer2D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(
        T_max=50,
        begin=0,
        by_epoch=True,
        end=50,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=3407)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/valPrivacyRaw.txt',
        data_prefix=dict(
            img=
            '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/data'
        ),
        filename_tmpl='{:06}.png',
        pipeline=[
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=25,
                test_mode=True,
                type='SampleFrames'),
            dict(io_backend='disk', type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='TenCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='RawframeDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True,
        type='SampleFrames'),
    dict(io_backend='disk', type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='TenCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=150, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=24,
    dataset=dict(
        ann_file=
        '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/trainPrivacyRaw.txt',
        data_prefix=dict(
            img=
            '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/data'
        ),
        filename_tmpl='{:06}.png',
        pipeline=[
            dict(
                clip_len=1, frame_interval=1, num_clips=6,
                type='SampleFrames'),
            dict(io_backend='disk', type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                input_size=224,
                max_wh_scale_gap=1,
                random_crop=False,
                scales=(
                    1,
                    0.875,
                    0.75,
                    0.5,
                ),
                type='MultiScaleCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='RawframeDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=1, frame_interval=1, num_clips=6, type='SampleFrames'),
    dict(io_backend='disk', type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        input_size=224,
        max_wh_scale_gap=1,
        random_crop=False,
        scales=(
            1,
            0.875,
            0.75,
            0.5,
        ),
        type='MultiScaleCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/valPrivacyRaw.txt',
        data_prefix=dict(
            img=
            '/home/node1/Desktop/code/ai/data/sbu/nc_plus_08size_3c_filter/data'
        ),
        filename_tmpl='{:06}.png',
        pipeline=[
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=6,
                test_mode=True,
                type='SampleFrames'),
            dict(io_backend='disk', type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='RawframeDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=6,
        test_mode=True,
        type='SampleFrames'),
    dict(io_backend='disk', type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/resnet_cls_softmax_privacy_08-zqf'
