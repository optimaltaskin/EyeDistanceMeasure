_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        ),
        point_head=dict(
            num_classes=1,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
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
log_config = dict(interval=4)
checkpoint_config = dict(interval=50)
runner = dict(type='EpochBasedRunner', max_epochs=100)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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

load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
