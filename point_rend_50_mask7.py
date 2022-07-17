_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
            loss_mask=dict(
                # Please try setting loss_weight=1.1
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        ),
        point_head=dict(
            num_classes=1,
            loss_point=dict(
                # Please try setting loss_weight=1.1
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ),
    # As long as memory is sufficient to complete testing phase after each epoch,
    # test_cfg parameter can be left commented.
    # test_cfg=dict(
    #     rpn=dict(
    #         max_per_img=100, ),
    #     rcnn=dict(
    #         max_per_img=10, ),
    # ),
)

dataset_type = 'COCODataset'
classes = ('card',)
# note: log_config interval sets frequency of logging progress of training. This value can be changes as wanted
log_config = dict(interval=4)
# note: checkpoint_config interval determines at which epochs, model will be saved. This value can be kept low to
# note: take a snapshot of progress more frequently. However, each model covers 100s MB of space.
checkpoint_config = dict(interval=2)
runner = dict(type='EpochBasedRunner', max_epochs=12)

# For optimizer please try each commented lines below separately:
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)


data = dict(
    # Please try following values for samples_per_gpu: [4, 8, 16]
    samples_per_gpu=2,
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

load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'
