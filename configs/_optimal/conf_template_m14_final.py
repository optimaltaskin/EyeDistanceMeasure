_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_roi_extractor=dict(
            _delete_=True,
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            _delete_=True,
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            num_classes=1,
            in_channels=1024,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            mask_size=14,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            max_per_img=100, ),
        rcnn=dict(
            max_per_img=10, ),
    ),
)
dataset_type = 'COCODataset'
classes = ('card',)
log_config = dict(interval=4, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=50)
runner = dict(type='EpochBasedRunner', max_epochs=50)

optimizer = dict(type='SGD', lr=0.0001, weight_decay=0.001)
# optimizer = dict(_delete_=True, type='AdamW', lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        # img_prefix='data/base_set/training/',
        img_prefix='data/base_set/training/',
        classes=classes,
        ann_file='data/base_set/training/training_labels.json'),
    test=dict(
        img_prefix='data/base_set/testing/',
        classes=classes,
        ann_file='data/base_set/testing/testing_labels.json'),
    val=dict(
        img_prefix='data/base_set/testing/',
        classes=classes,
        ann_file='data/base_set/testing/testing_labels.json'),
    # val=dict(
    #     img_prefix='data/base_set/validation/',
    #     classes=classes,
    #     ann_file='data/base_set/validation/val_labels.json'),
)

load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
