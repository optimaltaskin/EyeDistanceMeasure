_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'
model = dict(
    rpn_head=dict(
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            roi_layer=dict(type='RoIAlign', output_size=15, sampling_ratio=0),
        ),
            bbox_head=dict(
            num_classes=1,
        ),
        mask_roi_extractor=dict(
            # roi_layer=dict(type='SimpleRoIAlign', output_size=28),
            #     _delete_=True,
            #     type='SingleRoIExtractor',
            # out_channels=512,
            # roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            #     out_channels=256,
            #     featmap_strides=[4, 8, 16, 32]
        ),
        mask_head=dict(
            # num_convs=4,
            # in_channels=512,
            # _delete_=True,
            # type='FCNMaskHead',
            # num_convs=4,
            # in_channels=256,
            # conv_out_channels=256,
            num_classes=1,
            # loss_mask=dict(
            #     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        ),
        point_head=dict(
            num_classes=1,
            # in_channels=512,
            # in_channels=1024,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            mask_size=15,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            max_per_img=100,),
        rcnn=dict(
            max_per_img=10,),
    ),
)
dataset_type = 'COCODataset'
classes = ('card',)
log_config = dict(interval=4)
checkpoint_config = dict(interval=36)
# runner = dict(type='EpochBasedRunner', max_epochs=100)

# optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.00001)
optimizer = dict(lr=0.002)
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