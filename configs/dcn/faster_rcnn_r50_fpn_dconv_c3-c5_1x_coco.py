_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco__.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
