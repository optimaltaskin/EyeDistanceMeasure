_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
        ),
        point_head=dict(
            num_classes=1,
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
log_config = dict(interval=10)
checkpoint_config = dict(interval=2)
runner = dict(type='EpochBasedRunner', max_epochs=12)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(type='AdamW', lr=0.00001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        img_prefix='data/cocoset/training/images/',
        classes=classes,
        ann_file='data/cocoset/training/coco_instances.json'),
    test=dict(
        img_prefix='data/cocoset/testing/images/',
        classes=classes,
        ann_file='data/cocoset/testing/coco_instances.json'),
    val=dict(
        img_prefix='data/cocoset/testing/images/',
        classes=classes,
        ann_file='data/cocoset/testing/coco_instances.json'),
    # val=dict(
    #     img_prefix='data/base_set/validation/',
    #     classes=classes,
    #     ann_file='data/base_set/validation/val_labels.json'),
)

# load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
# load_from = 'checkpoints/pointrend_ResNeXt_model_final_ba17b9.pkl'
