_base_ = '../body/2d_kpt_sview_rgb_vid/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py'
# dataset settings
dataset_type = 'BottomUpCocoDataset'
classes = ('person')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # train=dict(
    #     ),
    # val=dict(
    #     ),
    test=dict(
        type=dataset_type,
        # ann_file='data/coco/annotations/COCO_labels.json',
        # img_prefix='data/coco/rgb/',
    ))
