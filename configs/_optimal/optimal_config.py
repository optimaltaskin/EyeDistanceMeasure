# The new config inherits a base config to highlight the necessary modification
_base_ = '../point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco_downloaded.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        type='PointRendRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            roi_layer=dict(type='SimpleRoIAlign', output_size=14),
            out_channels=256,
            featmap_strides=[4],
            aggregation='concat'),
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
        ),
        point_head=dict(
            num_classes=1,
        ),

    )
)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('card',)

data = dict(
    train=dict(
        img_prefix='data/base_set/training/',
        classes=classes,
        ann_file='data/base_set/training/training_labels.json'),
    val=dict(
        img_prefix='data/base_set/testing/',
        classes=classes,
        ann_file='data/base_set/testing/testing_labels.json'),
    test=dict(
        img_prefix='data/base_set/validation/',
        classes=classes,
        ann_file='data/base_set/validation/val_labels.json'),
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'