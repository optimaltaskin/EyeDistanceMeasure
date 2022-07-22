_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'
model = dict(
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
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=4)
runner = dict(type='EpochBasedRunner', max_epochs={EPOCH})
workflow = [('train', 1), ('val', 1)]
# optimizer = dict(type='SGD', lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})
optimizer = dict(_delete_=True, type='AdamW', lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        # img_prefix='data/base_set/training/',
        img_prefix='data/{SET_FOLDER}/training/images/',
        classes=classes,
        ann_file='data/{SET_FOLDER}/training/coco_instances.json'),
    test=dict(
        img_prefix='data/{SET_FOLDER}/testing/images/',
        classes=classes,
        ann_file='data/{SET_FOLDER}/testing/coco_instances.json'),
    val=dict(
        img_prefix='data/{SET_FOLDER}/validation/images/',
        classes=classes,
        ann_file='data/{SET_FOLDER}/validation/coco_instances.json'),
)

load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
