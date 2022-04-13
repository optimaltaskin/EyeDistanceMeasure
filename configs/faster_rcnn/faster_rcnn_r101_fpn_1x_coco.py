_base_ = './faster_rcnn_r50_fpn_1x_coco__.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
