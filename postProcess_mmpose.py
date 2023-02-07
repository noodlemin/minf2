import json
from collections import OrderedDict, defaultdict
import numpy as np
import pdb
import cv2
import os, csv
import datetime
from os import listdir

preds = []
scores = []
image_paths = []

def _sort_and_unique_bboxes(kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

def soft_oks_nms(kpts_db, thr, max_dets=20, sigmas=None, vis_thr=None):
    """Soft OKS NMS implementations.
    Args:
        kpts_db
        thr: retain oks overlap < thr.
        max_dets: max number of detections to keep.
        sigmas: Keypoint labelling uncertainty.
    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([k['score'] for k in kpts_db])
    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while len(order) > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        order = order[1:]
        scores = _rescore(oks_ovr, scores[1:], thr)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep

def _rescore(overlap, scores, thr, type='gaussian'):
    """Rescoring mechanism gaussian or linear.
    Args:
        overlap: calculated ious
        scores: target scores.
        thr: retain oks overlap < thr.
        type: 'gaussian' or 'linear'
    Returns:
        np.ndarray: indexes to keep
    """
    assert len(overlap) == len(scores)
    assert type in ['gaussian', 'linear']

    if type == 'linear':
        inds = np.where(overlap >= thr)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(-overlap**2 / thr)

    return scores


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.
    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious

def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_limb_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and limbs on an image.
    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_limb_color (np.array[Mx3]): Color of M limbs. If None, the
                limbs will not be drawn.
            thickness (int): Thickness of lines.
    """

    # img = mmcv.imread(img)
    img_h, img_w, _ = img.shape

    for kpts in pose_result:
        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                   radius, (int(r), int(g), int(b)), -1)
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        r, g, b = pose_kpt_color[kid]
                        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                                   (int(r), int(g), int(b)), -1)

        # draw limbs
        if skeleton is not None and pose_limb_color is not None:
            assert len(pose_limb_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                        and pos2[1] > 0 and pos2[1] < img_h
                        and kpts[sk[0] - 1, 2] > kpt_score_thr
                        and kpts[sk[1] - 1, 2] > kpt_score_thr):
                    r, g, b = pose_limb_color[sk_id]
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle), 0,
                            360, 1)
                        cv2.fillConvexPoly(img_copy, polygon,
                                           (int(r), int(g), int(b)))
                        transparency = max(
                            0,
                            min(
                                1, 0.5 *
                                (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(
                            img,
                            pos1,
                            pos2, (int(r), int(g), int(b)),
                            thickness=thickness)

    return img


def save_preds(path, valid_kpts, id = 0):
    project_name = path.split('/')[-1]

    pose_result = [valid_kpts[id][0]['keypoints']]
    # print(pose_result)
    image_id = valid_kpts[id][0]['image_id']

    filename = str(image_id).zfill(10)
    pose_result2 = []
    for i in [0,1,5,6,7,8,9,10]:
        if i == 1:
            pose_result2.append(tuple((pose_result[0][5]+pose_result[0][6])/2)[0:2])
        else:
            pose_result2.append(tuple((pose_result[0][i])[0:2]))
    joints = pose_result2

    datTim = datetime.datetime.now()
    if len(joints) == 8:
        with open(f'{path}/{project_name}_preds_HigherHRNet.csv','a', newline="") as p:
            row = [path+'/rgb/',filename,joints[0][:],joints[1][:],joints[3][:],joints[5][:],joints[7][:],joints[2][:],joints[4][:],joints[6][:],image_id,datTim.strftime("%x"),datTim.strftime("%X"), 'Pose Predictions']
            csv_out=csv.writer(p, lineterminator='\n')
            csv_out.writerow(row)
    else:
        with open(f'{path}/{project_name}_preds_HigherHRNet.csv','a', newline="") as p:
            row = [path+'/rgb/',filename,(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),image_id,datTim.strftime("%x"),datTim.strftime("%X"), 'Pose Predictions']
            csv_out=csv.writer(p, lineterminator='\n')
            csv_out.writerow(row)
    

def generate_preds(path, valid_kpts):
    rgb = os.path.join(path,'rgb')
    project_name = path.split('/')[-1]
    # print(path)
    # print(project_name)

    # with open(f'{path}/{project_name}_preds_HigherHRNet.csv','a', newline="") as p:
    with open(f'{path}/{project_name}_preds_HigherHRNet.csv','a', newline="") as p:
            row = ['path','imgName','head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist','image_id','date','time','Action']
            csv_out=csv.writer(p)
            csv_out.writerow(row)
    for i in range(len(valid_kpts)):
        save_preds(path,valid_kpts, id = i)



# path = '/mnt/c/Users/tkfps/Downloads/2d_poses/'
path = '/mnt/c/Users/tkfps/Downloads/2dposes/'
filelist = listdir(path)

# o = json.load(open(f'{path}/{filelist[1]}'))
# print(o[24])
# p = json.load(open('/mnt/d/dataset/20210523_202300/'+ filelist[1]))
# print(p[24])
# exit()
total_count = 0
for i in filelist:
    preds = []
    scores = []
    image_paths = []
    # project_name = path.split('/')[-1]
    project_name = i.split('_')[1] + '_' + i.split('_')[2]
    outpath = '/mnt/c/Users/tkfps/Downloads/2dpost/' + project_name
    # + project_name 
    # os.mkdir(outpath)
    # print(outpath)
    algos = 'bottomup'
    use_nms = 1
    # outputs = json.load(open(f'{outpath}/out_{project_name}_HigherHRNet.json'))
    output = json.load(open(f'{path}/{i}'))
    # print(output[0])
    # algos = 'bottomsup'
    # use_nms = 1
    # outputs = json.load(open('out.json'))
    print(i)
    if algos == 'bottomup':
        for j, out in enumerate(output):
            # print(out)
            preds.append(out[0]['keypoints'])
            scores.append(out[0]['score'])
            image_paths.append(str(j).zfill(10))
            # break
            kpts = defaultdict(list)
            # id2name, name2id = _get_mapping_id_name(coco.imgs)
            # iterate over images
            for idx, _preds in enumerate(preds):
                # str_image_path = image_paths[idx]
                # image_id = self.name2id[os.path.basename(str_image_path)]
                # image_id = int(os.path.basename(str_image_path).split('.')[0])
                # iterate over people
                if not _preds and preds[idx-1]:
                    _preds = preds[idx-1]      
                elif not _preds and preds[idx+1]:  
                    _preds = preds[idx+1]
                # use bbox area
                kpt = np.array(_preds)
                
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (
                        np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                
                kpts[idx].append({
                    'keypoints': kpt[:, :3],
                    'score': scores[idx],
                    'tags': kpt[:, :3],
                    'image_id': idx,
                    'area': area,
                })
                
            valid_kpts = []
            for img in kpts.keys():
                img_kpts = kpts[img]
                if use_nms:
                    nms = soft_oks_nms
                    keep = nms(img_kpts, 0.9, sigmas = np.array([
                        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                        .87, .87, .89, .89
                    ]) / 10.0)
                    valid_kpts.append([img_kpts[_keep] for _keep in keep])
                else:
                    valid_kpts.append(img_kpts)
        
    elif algos == 'topdown':
        kpts = defaultdict(list)
        for output in outputs:
            preds = np.array(output['preds'])
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = int(os.path.basename(image_paths[0]).split('.')[0])
                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = _sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = 17
        vis_thr = 0.2
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if use_nms:
                nms = soft_oks_nms
                keep = nms(img_kpts, 0.9, sigmas = np.array([
                    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
                    .87, .87, .89, .89
                ]) / 10.0)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)


    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])
# print('ahhhh:',total_count)
    generate_preds(outpath, valid_kpts)

# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
#                     [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
#                     [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# pose_limb_color = palette[[
#             0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
#         ]]
# pose_kpt_color = palette[[
#     16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
# ]]

# show_keypoint_weight = False
# # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (640,480))
# # pose_result = []
# # for valid_kpt in valid_kpts:
# #     pose_result.append(valid_kpt[0]['keypoints'])

# # total = 2000; n = 0; idxs = np.random.randint(0,14000,total).tolist();
# # for idx in idxs:
# #     pose_result = [valid_kpts[idx][0]['keypoints']]
# #     image_id = valid_kpts[idx][0]['image_id']

# #     img = cv2.imread(path+'/rgb/'+str(image_id).zfill(10)+'.png')    
# #     img_h, img_w, _ = img.shape

# #     skeleton = [[1,2],[2,3],[2,4],[4,6],[6,8],[3,5],[5,7]]
# #     pose_result2 = []


# #     if len(pose_result[0]) == 17:
# #         for i in [0,1,5,6,7,8,9,10]:
# #             if i == 1:
# #                 pose_result2.append(((pose_result[0][5]+pose_result[0][6])/2).tolist())
# #             else:
# #                 pose_result2.append(pose_result[0][i].tolist())
# #         pose_result2 = [np.array(pose_result2)]
# #         # print('Pose is 17')
# #         n = n + 1

# #     else:
# #         skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
# #                         [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
# #                         [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# #         pose_result2 = [pose_result[0]]
# #         # print(f'Pose is {len(pose_result[0])}')

# #     # pdb.set_trace()

# #     pose_limb_color = palette[[
# #                 5, 5, 0, 0, 7, 7, 5
# #             ]]
# #     pose_kpt_color = palette[[
# #         16, 16, 10, 16, 10, 14, 9, 9
# #     ]]
# #     img = imshow_keypoints(img,pose_result2,
# #                         skeleton=skeleton,
# #                         kpt_score_thr=0.3,
# #                         pose_kpt_color=pose_kpt_color,
# #                         pose_limb_color=pose_limb_color,
# #                         radius=4,
# #                         thickness=1,
# #                         show_keypoint_weight=False)

# #     # cv2.imshow('img', img)
# #     # cv2.waitKey()
# #     out.write(img)

# # print('Total 17 jointed detections', n/total*100)
# # out.release()

