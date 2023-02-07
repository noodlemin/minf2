# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import json
from os import listdir

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images."""
    # parser = ArgumentParser()
    # parser.add_argument('pose_config', help='Config file for pose')
    # parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    # parser.add_argument('--video-path', type=str, help='Video path')
    # parser.add_argument(
    #     '--show',
    #     action='store_true',
    #     default=False,
    #     help='whether to show visualizations.')
    # parser.add_argument(
    #     '--out-video-root',
    #     default='',
    #     help='Root of the output video file. '
    #     'Default not saving the visualization video.')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    # parser.add_argument(
    #     '--pose-nms-thr',
    #     type=float,
    #     default=0.9,
    #     help='OKS threshold for pose NMS')
    # parser.add_argument(
    #     '--radius',
    #     type=int,
    #     default=4,
    #     help='Keypoint radius for visualization')
    # parser.add_argument(
    #     '--thickness',
    #     type=int,
    #     default=1,
    #     help='Link thickness for visualization')

    # args = parser.parse_args()

    # assert args.show or (args.out_video_root != '')
    
    # build the pose model from a config file and a checkpoint file
    # pose_model = init_pose_model(
        # args.pose_config, args.pose_checkpoint, device=args.device.lower())
    config = 'configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py'
    check_point = 'https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth'

    pose_model = init_pose_model(config, check_point, 'mps')

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video_path = '/Users/minseongkim/minf2/a/'

    list = listdir(video_path)
    for i in list:
        name = i.split('.')[0]
        print(name)
        video = mmcv.VideoReader(video_path+i)
        assert video.opened, f'Faild to load video file {video_path+i}'

        # if out_video_root == '':
        #     save_out_video = False
        # else:
        #     os.makedirs(args.out_video_root, exist_ok=True)
        #     save_out_video = True

        # if save_out_video:
        #     fps = video.fps
        #     size = (video.width, video.height)
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     videoWriter = cv2.VideoWriter(
        #         os.path.join(args.out_video_root,
        #                     f'vis_{os.path.basename(args.video_path)}'), fourcc,
        #         fps, size)

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        print('Running inference...')
        
        poses = []
        posess = []
        for _, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            pose = {}
            pose_results, _ = inference_bottom_up_pose_model(
                pose_model,
                cur_frame,
                dataset=dataset,
                dataset_info=dataset_info,
                pose_nms_thr=0.9,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            
            for i in pose_results:
                i['keypoints'] = i['keypoints'].tolist()
            poses.append(pose_results)
            
            # show the results
        #     vis_frame = vis_pose_result(
        #         pose_model,
        #         cur_frame,
        #         pose_results,
        #         radius=args.radius,
        #         thickness=args.thickness,
        #         dataset=dataset,
        #         dataset_info=dataset_info,
        #         kpt_score_thr=args.kpt_thr,
        #         show=False)

        #     if args.show:
        #         cv2.imshow('Image', vis_frame)

        #     if save_out_video:
        #         videoWriter.write(vis_frame)

        #     if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # if save_out_video:
        #     videoWriter.release()
        # if args.show:
        #     cv2.destroyAllWindows()
        # print(poses)
        
        
        # outpath = '/home/tkfpsk/minf2/output/fake_pose/'
        outpath = '/Users/minseongkim/minf2/'
        pj = str(poses)
        result = json.dumps(pj)
        result = result.replace('"', '')
        result = result.replace("'", '"')
        # print(result)
        # name = os.path.basename(video_path)
        # name = name.split('.')[0]
        # print(name)
        # name = str(i)
        # print(name)
        with open(outpath + 'out_' + name + '_HigherHRNet.json', "w") as f:
            f.write(result)
if __name__ == '__main__':
    main()
