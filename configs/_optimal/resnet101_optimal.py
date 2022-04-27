_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco_downloaded.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.1)),
        point_head=dict(
            num_classes=1,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.1))),
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
            scale_factor=2))
)

dataset_type = 'COCODataset'
classes = ('card',)
log_config = dict(interval=4)
checkpoint_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=64)

# optimizer = dict(type='Adam', lr=0.0002, weight_decay=0.0001)
optimizer = dict(lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
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

# load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth'