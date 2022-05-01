_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
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
            fc_channels=1024,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            mask_size=56,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=10,
            mask_thr_binary=0.5,
            subdivision_steps=5,
            subdivision_num_points=784,
            scale_factor=2)),
)
dataset_type = 'COCODataset'
classes = ('card',)
log_config = dict(interval=4)
checkpoint_config = dict(interval=18)
runner = dict(type='EpochBasedRunner', max_epochs=18)

optimizer = dict(type='AdamW', lr=1e-05, weight_decay=5e-06)
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        # img_prefix='data/base_set/training/',
        img_prefix='data/scaled_populated/training/',
        classes=classes,
        ann_file='data/scaled_populated/training/training_labels.json'),
    test=dict(
        img_prefix='data/scaled_populated/testing/',
        classes=classes,
        ann_file='data/scaled_populated/testing/testing_labels.json'),
    val=dict(
        img_prefix='data/scaled_populated/testing/',
        classes=classes,
        ann_file='data/scaled_populated/testing/testing_labels.json'),
    # val=dict(
    #     img_prefix='data/base_set/validation/',
    #     classes=classes,
    #     ann_file='data/base_set/validation/val_labels.json'),
)

load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'