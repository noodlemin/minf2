import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import pandas as pd
import pdb
# from udp.body import Body
import cv2
import seaborn as sns
from scipy.ndimage import gaussian_filter, sobel
from scipy import stats
from scipy.interpolate import CubicSpline, Rbf, interp2d
from scipy.special import rel_entr, kl_div
import datetime, csv
import copy
import pyrealsense2 as rs

# from KalmanFilter import KalmanFilter, KalmanFilter1D
# import statsmodels.api as sm
from math import log2, log

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from sklearn.metrics import mean_squared_error
 
def ed_error(preds,gts):
    # retruns a vector of |x-x_hat|^2
    # ed = np.linalg.norm((np.array(preds)-np.array(gts)))
    try:
        ed = np.sqrt(np.sum((np.array(preds)-np.array(gts))**2, axis=1))
    except:
        ed = np.sqrt((np.array(preds)-np.array(gts))**2)
    return ed

def ed_error2(preds,gts):
    # retruns a vector of |x-x_hat|^2
    # ed = np.linalg.norm((np.array(preds)-np.array(gts)))
    ed = np.sum((preds - gts)**2)/(np.shape(preds)[0]*np.shape(preds)[1])
    return ed

def proj2Dto3D(pts2D, depImg, pts3Dmet = True, dilt = True):
    # project 2D points to 3D with depth map
    pts3D = []
    depImg_orig = copy.deepcopy(depImg)
    kernel = np.ones((6,6), np.uint8)
    if dilt == True:
        depImg = cv2.dilate(depImg, kernel, cv2.BORDER_REFLECT) 
    else:
        depImg = depImg
    # pdb.set_trace()

    # fig, [ax1,ax2] = plt.subplots(1,2)
    # ax1.matshow(depImg_orig)
    # ax2.matshow(depImg)
    # plt.show()
    # pdb.set_trace()

    # sns.heatmap(depImg_orig, annot=True, ax=ax1)
    # sns.heatmap(depImg, annot=True, ax=ax2)
    # plt.show()

    for i in range(len(pts2D)):
        x = int(pts2D[i][0]); y = int(pts2D[i][1])
        if pts3Dmet == True:
            result = rs.rs2_deproject_pixel_to_point(make_intrinsics(), [x, y], depImg[(y,x)])
            pts3D.append((-result[0], -result[1], result[2]))
        else:
            pts3D.append((x,y,depImg[(y,x)]))

    return pts3D

def get2Dskeleton(path, id = 0, dep = True, gt=True, csvI=False, csvName= None):
    project_name = path.split('/')[-1];flag = 0
    if csvName is not None:
        csvfile = path +'/'+ csvName
    elif (csvName is None) and (gt is True):
        csvfile = path +'/'+ project_name + '.csv'
    elif (csvName is None) and (gt is False) and (csvI is True):
        csvfile = path +'/'+ project_name + '_preds_udp.csv'

    if gt == True:
        df = pd.read_csv(csvfile)
        df = df.sort_values(by=['imgName'])
        row = df.loc[df['imgName'].values == id]
        filename = str(int(row['imgName'])).zfill(10)
        pts2D = [(eval(row['head'].values[0])[0],eval(row['head'].values[0])[1]),
                    (int((eval(row['Left-shoulder'].values[0])[0]+eval(row['Right-Shoulder'].values[0])[0])/2),int((eval(row['Left-shoulder'].values[0])[1]+eval(row['Right-Shoulder'].values[0])[1])/2)),
                    (eval(row['Right-Shoulder'].values[0])[0],eval(row['Right-Shoulder'].values[0])[1]),
                    (eval(row['Right-Elbow'].values[0])[0],eval(row['Right-Elbow'].values[0])[1]),
                    (eval(row['Right-Wrist'].values[0])[0],eval(row['Right-Wrist'].values[0])[1]),
                    (eval(row['Left-shoulder'].values[0])[0],eval(row['Left-shoulder'].values[0])[1]),
                    (eval(row['Left-Elbow'].values[0])[0],eval(row['Left-Elbow'].values[0])[1]),
                    (eval(row['Left-Wrist'].values[0])[0],eval(row['Left-Wrist'].values[0])[1]),
                    ]
    elif gt == False and csvI == True:
        df = pd.read_csv(csvfile)
        df = df.sort_values(by=['imgName'])
        row = df.loc[df['imgName'].values == id]
        filename = str(int(row['imgName'])).zfill(10)
        pts2D = [(eval(row['head'].values[0])[0],eval(row['head'].values[0])[1]),
                    (int((eval(row['Left-shoulder'].values[0])[0]+eval(row['Right-Shoulder'].values[0])[0])/2),int((eval(row['Left-shoulder'].values[0])[1]+eval(row['Right-Shoulder'].values[0])[1])/2)),
                    (eval(row['Right-Shoulder'].values[0])[0],eval(row['Right-Shoulder'].values[0])[1]),
                    (eval(row['Right-Elbow'].values[0])[0],eval(row['Right-Elbow'].values[0])[1]),
                    (eval(row['Right-Wrist'].values[0])[0],eval(row['Right-Wrist'].values[0])[1]),
                    (eval(row['Left-shoulder'].values[0])[0],eval(row['Left-shoulder'].values[0])[1]),
                    (eval(row['Left-Elbow'].values[0])[0],eval(row['Left-Elbow'].values[0])[1]),
                    (eval(row['Left-Wrist'].values[0])[0],eval(row['Left-Wrist'].values[0])[1]),
                    ]
    else:
        filename = str(id).zfill(10)
        img = cv2.imread(path+'/rgb/'+filename+'.png')
        pts2D = []
        # pts2D = process_frame_openpose(img, body=True)

    pts_dummy = [list(pt) for pt in pts2D]
    if np.all(np.array(pts_dummy) == 0):
        print(id,',')
        flag = 1
    return pts2D, filename

def get3Dskeleton(path, id = 0, dep = True, gt=True, csvI = False, csvName= None, pts3Dmet = True, dilt = True):


    pts2D, filename = get2Dskeleton(path=path, id = id, dep = dep, gt=gt, csvI=csvI, csvName= csvName)

    if dep == True and pts2D:
        dep = os.path.join(path,'depth')
        depImg = np.load(dep+'/'+filename+'.npy')
        pts3D = proj2Dto3D(pts2D, depImg, pts3Dmet = pts3Dmet, dilt = dilt)
    else:
        # print('3D skeleton needs 3D depth maps OR 2D pts dont exist')
        pts3D = []            

    return pts3D, filename

def vis2Dskeleton(path, pts2D, filename):
    img = cv2.imread(path+'/rgb/'+filename+'.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    joints = [[0,1],[2,3],[3,4],[5,2],[5,6],[6,7]]
    fig = plt.figure()
    xs=[];ys=[]
    ax = plt.axes()
    for i in range(len(pts2D)):
        xs.append(pts2D[i][0])
        ys.append(pts2D[i][1])

    try:
        for j in joints:
            ax.plot([xs[j[0]], xs[j[1]]], [ys[j[0]], ys[j[1]]])
        
        ax.imshow(img)
        plt.show()
    except:
        print(f'Size of pts2D: {pts2D}')

def vis2Dskeleton_depth(path, pts2D, filename):
    dep = os.path.join(path,'depth')
    depImg = np.load(dep+'/'+filename+'.npy')
    # depImg = cv2.cvtColor(depImg, cv2.COLOR_BGR2RGB)
    joints = [[0,1],[2,3],[3,4],[5,2],[5,6],[6,7]]
    fig = plt.figure()
    xs=[];ys=[]
    ax = plt.axes()
    for i in range(len(pts2D)):
        xs.append(pts2D[i][0])
        ys.append(pts2D[i][1])

    try:
        for j in joints:
            ax.plot([xs[j[0]], xs[j[1]]], [ys[j[0]], ys[j[1]]])
        # ax.matshow(depImg)
        ax.imshow(depImg)
        # for i in range(len(pts2D)):
        #     print(xs[i],ys[i], depImg[(int(xs[i]),int(ys[i]))], depImg[(int(ys[i]),int(xs[i]))])
        # pdb.set_trace()
        plt.show()
    except:
        print(f'Size of pts2D: {pts2D}')

def vis2Dskeleton_overlay(path, pts2D_gt, pts2D_pred, filename):
    img = cv2.imread(path+'/rgb/'+filename+'.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    joints = [[0,1],[2,3],[3,4],[5,2],[5,6],[6,7]]
    fig = plt.figure()
    xs=[];ys=[];xs_p=[];ys_p=[]
    ax = plt.axes()
    for i in range(len(pts2D_gt)):
        xs.append(pts2D_gt[i][0])
        ys.append(pts2D_gt[i][1])
        xs_p.append(pts2D_pred[i][0])
        ys_p.append(pts2D_pred[i][1])

    try:
        for j in joints:
            ax.plot([xs[j[0]], xs[j[1]]], [ys[j[0]], ys[j[1]]],'b',linewidth=5)
        for p in joints:
            ax.plot([xs_p[p[0]], xs_p[p[1]]], [ys_p[p[0]], ys_p[p[1]]], 'g', linewidth=3)
        
        ax.imshow(img)
        plt.show()
    except:
        print(f'Size of pts2D: {pts2D_gt}')

def vis3Dskeleton(pts3D):
    # joints = [[1,2],[1,3],[3,5],[2,4],[4,6],[0,7]]
    joints = [[0,1],[2,3],[3,4],[5,2],[5,6],[6,7]]
    fig = plt.figure()
    xs=[];ys=[];zs=[];
    ax = plt.axes(projection='3d')
    for i in range(len(pts3D)):
        xs.append(pts3D[i][0])
        ys.append(pts3D[i][1])
        zs.append(pts3D[i][2])

    for j in joints:
        ax.plot([xs[j[0]], xs[j[1]]], [ys[j[0]], ys[j[1]]], [zs[j[0]], zs[j[1]]])

    ax.scatter(xs,ys,zs)
    # print(pts3D)
    plt.show()

def temporal_3D_flow(df_pred, df_gt, df3 = None, lim=[0,4.5], strng='Title'):

    limb = ['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist']
    # limb = ['head_z','Mid-Shoulder_z','Right-Shoulder_z','Right-Elbow_z','Right-Wrist_z','Left-shoulder_z','Left-Elbow_z','Left-Wrist_z']
    fig, ax = plt.subplots(2,4)
    fig.set_size_inches(18.5, 10.5)
    fig.set_dpi(100)
    ax = ax.ravel()
    # sns.boxplot(data = df_gt, ax=ax1);  sns.boxplot(data = df_pred, ax=ax2);
    # sns.boxplot(data=limbs_pred)
    # sns.boxplot(data=limbs_gt)
    # sns.set_color_codes("dark")
    if df3 is None:
        for i in range(len(limb)):
            dfGT = df_gt[limb[i]].reset_index(drop=True)
            dfpred = df_pred[limb[i]].reset_index(drop=True)
            sns.lineplot(data=dfGT, ax=ax[i], color="r")
            sns.lineplot(data=dfpred, ax=ax[i],color="g")
            ax[i].set_ylim([lim[0],lim[1]])

    else:
        for i in range(len(limb)):
            dfGT = df_gt[limb[i]].reset_index(drop=True)
            dfpred = df_pred[limb[i]].reset_index(drop=True)
            df3PP = df3[limb[i]].reset_index(drop=True)
            sns.lineplot(data=dfGT, ax=ax[i],color="r")
            sns.lineplot(data=dfpred,ax=ax[i], color="g")
            sns.lineplot(data=df3PP,ax=ax[i], color="b")
            ax[i].set_ylim([lim[0],lim[1]])
        
        # ax[i].plot(limbs_pred[list(limbs_pred.keys())[i]], 'b')
        # ax[i].plot(limbs_gt[list(limbs_gt.keys())[i]], 'r')
        # ax[i].set_title(list(limbs_gt.keys())[i])
    fig.suptitle(strng)
    # fig.savefig(f'Pred_{strng}_VS_GT.png', dpi=100)
    # plt.show()

def boxplot2D(df, ax):
    dfX = df[0::3];dfY=df[1::3];dfZ=df[2::3];inliers=[];whis=1.5
    # for i in [0,1,2]:
    #     dfTemp = df[i::3]
    #     Q1 = dfTemp.quantile(0.25)
    #     Q3 = dfTemp.quantile(0.75)
    #     IQR = Q3 - Q1
    #     total_inliers = ((dfTemp > (Q1 - 1.5 * IQR)) | (dfTemp < (Q3 + 1.5 * IQR)))
    #     inliers.append(total_inliers)
    limbs = ['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist']
    x1 = dfX.quantile(0.25);x2 = dfX.quantile(0.50);x3 = dfX.quantile(0.75)
    y1 = dfY.quantile(0.25);y2 = dfY.quantile(0.50);y3 = dfY.quantile(0.75)

    for limb in limbs:
        x = np.array(dfX[limb].tolist()); y = np.array(dfY[limb].tolist());
        xlimits = [x1[limb], x2[limb],x3[limb]]
        ylimits = [y1[limb], y2[limb],y3[limb]]

        box = Rectangle(
            (xlimits[0],ylimits[0]),
            (xlimits[2]-xlimits[0]),
            (ylimits[2]-ylimits[0]),
            ec = 'g',
            zorder=0
        )
        ax.add_patch(box)

        ##the x median
        vline = Line2D(
            [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
            color='g',
            zorder=1
        )
        ax.add_line(vline)

        ##the y median
        hline = Line2D(
            [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
            color='g',
            zorder=1
        )
        ax.add_line(hline)
        # # ax.plot([xlimits[1]],[ylimits[1]], color='g', marker='o')

        iqr = xlimits[2]-xlimits[0]
        ##left
        left = np.min(x[x >= xlimits[0]-whis*iqr])
        whisker_line = Line2D(
            [left, xlimits[0]], [ylimits[1],ylimits[1]],
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [left, left], [ylimits[0],ylimits[2]],
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_bar)  

        ##right
        right = np.max(x[x <= xlimits[2]+whis*iqr])
        whisker_line = Line2D(
            [right, xlimits[2]], [ylimits[1],ylimits[1]],
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [right, right], [ylimits[0],ylimits[2]],
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_bar)

        ##the y-whisker
        iqr = ylimits[2]-ylimits[0]

        ##bottom
        bottom = np.min(y[y >= ylimits[0]-whis*iqr])
        whisker_line = Line2D(
            [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [xlimits[0],xlimits[2]], [bottom, bottom], 
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_bar)
        ##top
        top = np.max(y[y <= ylimits[2]+whis*iqr])
        whisker_line = Line2D(
            [xlimits[1],xlimits[1]], [top, ylimits[2]], 
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [xlimits[0],xlimits[2]], [top, top], 
            color = 'g',
            zorder = 1
        )
        ax.add_line(whisker_bar)

        ##outliers
        mask = (x<left)|(x>right)|(y<bottom)|(y>top)
        ax.scatter(
            x[mask],y[mask],
            facecolors='none', edgecolors='g'
        )                       

def outliers_count(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    total_outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum().sum()
    total_elements = df.count().sum()
    outlier_elements = (total_outliers / total_elements) * 100
    outlier_imageSample = (df.loc[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)].shape[0] / df.shape[0])*100
    print(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum())
    return outlier_elements, outlier_imageSample

def zerofiller(dfX, dfY, dfZ):
    df_temp = copy.deepcopy(df)
    dfX = df[0::3]
    dfY = df[1::3]
    dfZ = df[2::3]
    idx = dfX.index[(dfX == 0).all(axis=1)]
    idy = dfY.index[(dfY == 0).all(axis=1)]
    idz = dfZ.index[(dfZ == 0).all(axis=1)]
    window = 3

    if len(idx)!=0:
        for ids in idx:
            dfX.loc[ids] = np.mean(dfX.loc[ids-3*window:ids+3*window], axis=0)
            # dfX.loc[ids] = np.median(dfX.loc[ids-4:ids+4], axis=0)

    if len(idy)!=0:
        for ids in idy:
            dfY.loc[ids] = np.mean(dfY.loc[ids-3*window:ids+3*window], axis=0)
            # dfY.loc[ids] = np.median(dfY.loc[ids-4:ids+4], axis=0)

    if len(idz)!=0:
        for ids in idz:
            dfZ.loc[ids] = np.mean(dfZ.loc[ids-3*window:ids+3*window], axis=0)
            # dfZ.loc[ids] = np.median(dfZ.loc[ids-4:ids+4], axis=0)

    df_temp[0::3] = dfX
    df_temp[1::3] = dfY
    df_temp[2::3] = dfZ

    return df_temp

def interpolation(pts3D):
    xs=np.zeros(8);ys=np.zeros(8);zs=np.zeros(8);
    for i in range(len(pts3D)):
        xs[i] = pts3D[i][0]
        ys[i] = pts3D[i][1]
        zs[i] = pts3D[i][2]
    

    if len(zs[zs > 3]) > 0:
        idxs = np.where(zs > 3)[0]
        notidxs = np.where(zs <= 3)[0]
    elif len(zs[zs < 0.2]) > 2:
        idxs = np.where(zs <= 1)[0]
        notidxs = np.where(zs > 1)[0]

    intrp = interp2d(xs[notidxs], ys[notidxs], zs[notidxs], kind='linear')
    # intrp = Rbf(xs[notidxs], ys[notidxs], zs[notidxs], function='multiquadric')
    for ids in idxs:
        zs[ids] = intrp(xs[ids], ys[ids])

    return np.hstack((xs,ys,zs)).reshape((3,8)).T

def interpdf(df):
    df_temp = copy.deepcopy(df)
    dfZ = df[2::3]
    limbs = ['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist']
    window = 2
    for limb in limbs:
        idz = dfZ[limb].index[(dfZ[limb] > dfZ[limb].median()+0.05) | (dfZ[limb] < dfZ[limb].median() - 0.05)]

        if len(idz)!=0:
            for ids in idz:
                # if (dfZ[limb].loc[ids-3*window:ids+3*window] > dfZ[limb].median()+0.05).all():
                #     dfZ[limb][ids-3*(window+1):ids+3*(window+1)] = dfZ[limb].median()
                # else:
                dfZ[limb][ids] = np.mean(dfZ[limb].loc[ids-3*window:ids+3*window], axis=0)
                # dfZ.loc[ids] = np.median(dfZ.loc[ids-4:ids+4], axis=0)
                # dfZ[limb][ids] = np.mean(dfZ[limb].loc[ids-3*window:ids+3*window], axis=0)

    df_temp[2::3] = dfZ

    return df_temp

def make_intrinsics():
    '''
    Avoid having to read a bagfile to get the camera intrinsics
    '''
    # Copied from a bagfile's intrinsics
    intrinsics = rs.intrinsics()
    intrinsics.coeffs = [0,0,0,0,0]
    intrinsics.fx = 617.916
    intrinsics.fy = 617.453
    intrinsics.height = 480
    intrinsics.ppx = 328.326
    intrinsics.ppy = 247.22
    intrinsics.width=640
    return intrinsics
    
    

# def process_frame_udp(frame, body=True):

#     def decode(candidate, subset):
#         pts = []
#         idxs = [0, 1, 2, 3, 4, 5, 6, 7]
#         for i in range(18):
#             for n in range(len(subset)):
#                 index = int(subset[n][i])
#                 if index == -1 or index not in idxs:
#                     continue
#                 x, y = candidate[index][0:2]
#                 pts.append((int(x),int(y)))
#         return pts

#     body_estimation = Body('D:\PhD Edinburgh\Codes\labellers\labeller\src\udp\model/body_pose_model.pth')
#     # canvas = copy.deepcopy(frame)
#     if body:
#         candidate, subset = body_estimation(frame)
#         # canvas = util.draw_bodypose2(canvas, candidate, subset)
#         # canvas = util.draw_handpose(canvas, all_hand_peaks)
#     pts = decode(candidate, subset)
#     return pts

def save_preds(path, id = 0):
    project_name = path.split('/')[-1]
    filename = str(id).zfill(10)
    img = cv2.imread(path+'/rgb/'+filename+'.png')
    # pts2D = process_frame_udp(img, body=True)
    joints = pts2D
    datTim = datetime.datetime.now()
    if len(pts2D) == 8:
        with open(f'{path}/{project_name}_preds.csv','a', newline="") as p:
            row = [path+'/rgb/',filename,joints[0][:],joints[1][:],joints[2][:],joints[3][:],joints[4][:],joints[5][:],joints[6][:],joints[7][:],id,datTim.strftime("%x"),datTim.strftime("%X"), 'Pose Predictions']
            csv_out=csv.writer(p, lineterminator='\n')
            csv_out.writerow(row)
    else:
        with open(f'{path}/{project_name}_preds.csv','a', newline="") as p:
            row = [path+'/rgb/',filename,(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),id,datTim.strftime("%x"),datTim.strftime("%X"), 'Pose Predictions']
            csv_out=csv.writer(p, lineterminator='\n')
            csv_out.writerow(row)

def generate_preds(path):
    rgb = os.path.join(path,'rgb')
    rgbImgs = os.listdir(rgb)
    project_name = path.split('/')[-1]
    with open(f'{path}/{project_name}_preds.csv','a', newline="") as p:
            # p.write('path,imgName,head,Mid-Shoulder,Right-Shoulder,Right-Elbow,Right-Wrist,Left-shoulder,Left-Elbow,Left-Wrist,image_id,date,time')
            row = ['path','imgName','head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist','image_id','date','time','Action']
            csv_out=csv.writer(p)
            csv_out.writerow(row)
    for i in range(len(rgbImgs)):
        save_preds(path, id = i)

def paired_ttest(df_pred, df_gt):
    limbs = ['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist']
    for limb in limbs:
        if (stats.shapiro(df_pred[limb])[1] > 0.05 and stats.shapiro(df_gt[limb])[1] > 0.05) or stats.shapiro(df_pred[limb]-df_gt[limb])[1] > 0.05:
            print(f'T test (paired) for {limb}: ',stats.ttest_rel(df_gt[limb], df_pred[limb]))
        else:
            print(f'The distribution of {limb} is not Normal/Gaussian. Value calculated with wilcoxon signed is: \n {stats.wilcoxon(df_gt[limb], df_pred[limb])}\n')

def kl_divergence(df_pred, df_gt):
    limbs = ['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist']
    for limb in limbs:
        p = np.array(df_gt[limb][2::3].tolist()); q = np.array(df_pred[limb][2::3].tolist())
        # print(f'kl divergence for {limb}: ',np.sum(np.where(p != 0, p * np.log(p / q), 0)))
        # print(f'kl divergence for {limb} (scipy-re_entr): ',sum(rel_entr(p,q)))
        print(f'kl divergence for {limb} (scipy-kl_div): ',sum(kl_div(p,q)))
        # print(f'kl divergence for {limb} (scipy-Entropy): ',sum(stats.entropy(p,q)),stats.entropy(p,q))

def kfInit(kf1D = True, kf2D=True):
    std_acc=50; x_std_meas=0.02; y_std_meas=0.024;std_acc_1D=50; x_std_1D=0.24;

    if kf1D == True:
        KF1D_head = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        KF1D_midS = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        KF1D_RS = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        KF1D_RE = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        KF1D_RW = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        KF1D_LS = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        KF1D_LE = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        KF1D_LW = KalmanFilter1D(1/30, 1, std_acc_1D, x_std_1D)
        kf1 = [KF1D_head, KF1D_midS, KF1D_RS,KF1D_RE, KF1D_RW, KF1D_LS, KF1D_LE, KF1D_LW]
    else:
        kf1 = []

    if kf2D == True:
        KF_head = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        KF_midS = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        KF_RS = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        KF_RE = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        KF_RW = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        KF_LS = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        KF_LE = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        KF_LW = KalmanFilter(1/30, 1, 1, std_acc, x_std_meas, y_std_meas)
        kf2 = [KF_head, KF_midS, KF_RS, KF_RE, KF_RW, KF_LS, KF_LE, KF_LW]
    else:
        kf2 = []
    
    return [kf1, kf2]


def kf(pts3D, kfinit, kf1D = True, kf2D=True):
    if len(kfinit[0]) > 0 and len(kfinit[1]) > 0:
        [[KF1D_head, KF1D_midS, KF1D_RS,KF1D_RE, KF1D_RW, KF1D_LS, KF1D_LE, KF1D_LW], [KF_head, KF_midS, KF_RS, KF_RE, KF_RW, KF_LS, KF_LE, KF_LW]] = kfinit
    elif len(kfinit[0]) > 0 and len(kfinit[1]) == 0:
        [[KF1D_head, KF1D_midS, KF1D_RS,KF1D_RE, KF1D_RW, KF1D_LS, KF1D_LE, KF1D_LW], kf2] = kfinit
    elif len(kfinit[0]) == 0 and len(kfinit[1]) > 0:
        [kf1, [KF_head, KF_midS, KF_RS, KF_RE, KF_RW, KF_LS, KF_LE, KF_LW]] = kfinit
    if kf1D == True:   
        x1D_head = KF1D_head.predict() 
        x11D_head = KF1D_head.update(pts3D[0,2])
        x1D_midS = KF1D_midS.predict() 
        x11D_midS = KF1D_midS.update(pts3D[1,2])
        x1D_RS = KF1D_RS.predict() 
        x11D_RS = KF1D_RS.update(pts3D[2,2])
        x1D_RE = KF1D_RE.predict() 
        x11D_RE = KF1D_RE.update(pts3D[3,2])
        x1D_RW = KF1D_RW.predict() 
        x11D_RW = KF1D_RW.update(pts3D[4,2])
        x1D_LS = KF1D_LS.predict() 
        x11D_LS = KF1D_LS.update(pts3D[5,2])
        x1D_LE = KF1D_LE.predict() 
        x11D_LE = KF1D_LE.update(pts3D[6,2])
        x1D_LW = KF1D_LW.predict() 
        x11D_LW = KF1D_LW.update(pts3D[7,2]) 

    if kf2D == True:
        x_head = KF_head.predict()
        x1_head = KF_head.update((pts3D[0,0], pts3D[0,1]))
        x_midS = KF_midS.predict()
        x1_midS = KF_midS.update((pts3D[1,0], pts3D[1,1]))
        x_RS = KF_RS.predict()
        x1_RS = KF_RS.update((pts3D[2,0], pts3D[2,1]))
        x_RE = KF_RE.predict()
        x1_RE = KF_RE.update((pts3D[3,0], pts3D[3,1]))
        x_RW = KF_RW.predict()
        x1_RW = KF_RW.update((pts3D[4,0], pts3D[4,1]))
        x_LS = KF_LS.predict()
        x1_LS = KF_LS.update((pts3D[5,0], pts3D[5,1]))
        x_LE = KF_LE.predict()
        x1_LE = KF_LE.update((pts3D[6,0], pts3D[6,1]))
        x_LW = KF_LW.predict()
        x1_LW = KF_LW.update((pts3D[7,0], pts3D[7,1]))


    if kf1D == True and kf2D == True:
        KF3D = np.array(
        [
                [x1_head[0,0], x1_head[0,1],x11D_head[0,0]],
                [x1_midS[0,0], x1_midS[0,1],x11D_midS[0,0]],
                [x1_RS[0,0], x1_RS[0,1],x11D_RS[0,0]],
                [x1_RE[0,0], x1_RE[0,1],x11D_RE[0,0]],
                [x1_RW[0,0], x1_RW[0,1],x11D_RW[0,0]],
                [x1_LS[0,0], x1_LS[0,1],x11D_LS[0,0]],
                [x1_LE[0,0], x1_LE[0,1],x11D_LE[0,0]],
                [x1_LW[0,0], x1_LW[0,1],x11D_LW[0,0]],
        ])
    elif kf1D == True and kf2D == False:
        KF3D = np.array(
        [
                [pts3D[0,0], pts3D[0,1],x11D_head[0,0]],
                [pts3D[1,0], pts3D[1,1],x11D_midS[0,0]],
                [pts3D[2,0], pts3D[2,1],x11D_RS[0,0]],
                [pts3D[3,0], pts3D[3,1],x11D_RE[0,0]],
                [pts3D[4,0], pts3D[4,1],x11D_RW[0,0]],
                [pts3D[5,0], pts3D[5,1],x11D_LS[0,0]],
                [pts3D[6,0], pts3D[6,1],x11D_LE[0,0]],
                [pts3D[7,0], pts3D[7,1],x11D_LW[0,0]],
        ])
    elif kf1D == False and kf2D == True:
        KF3D = np.array(
        [
                [x1_head[0,0], x1_head[0,1],pts3D[0,0]],
                [x1_midS[0,0], x1_midS[0,1],pts3D[1,0]],
                [x1_RS[0,0], x1_RS[0,1],pts3D[2,0]],
                [x1_RE[0,0], x1_RE[0,1],pts3D[3,0]],
                [x1_RW[0,0], x1_RW[0,1],pts3D[4,0]],
                [x1_LS[0,0], x1_LS[0,1],pts3D[5,0]],
                [x1_LE[0,0], x1_LE[0,1],pts3D[6,0]],
                [x1_LW[0,0], x1_LW[0,1],pts3D[7,0]],
        ])
    return KF3D

def VisualizeError(paths):
    eds2D = np.empty([8,1]); eds3D = np.empty([8,1]); missed_count=0;n=0;q = 0;outliers_count3D=0;outliers_count2D=0;tots=0;failed_count_2D=0;failed_count_3D=0
    pts_full_pred = np.empty([8,1]); pts_full_gt = np.empty([8,1]);pts_KF = np.empty([8,1]); limbs_2D = {}; limbs_3D = {};limbs_3D_prev={};limbs_2D_prev={};
    limbs_3D_KF = {}; limbs_2D_KF = {}; dilt = False; pts3Dmet = True; dff_pos=np.ones([8,3]);mn_pos=[];mn_vel=[]; dff_vel=np.ones([8,3]);holes=0;holes_gt=0;

    for path in paths:
        n = n + 1; q = 0
        rgb = os.path.join(path,'rgb')
        rgbImgs = os.listdir(rgb)

        kfinit = kfInit(kf1D = True, kf2D=True)

        diff_imgs = 0
        # sampld = np.random.randint(0+diff_imgs, int(len(rgbImgs))-diff_imgs,  size=int(len(rgbImgs)/(3))).tolist() #sampling 33% of ids from a folder of data randomly
        sampld = range(0,int(len(rgbImgs)/3))
        # sampld = np.random.randint(0+diff_imgs, int(len(rgbImgs))-diff_imgs,  size=int(len(rgbImgs))).tolist() #sampling 33% of ids from a folder of data randomly
        tots = tots + len(sampld)
        for i in sampld:
            q = q + 1;here=0
            # img = cv2.imread(rgb+imgNames)
            try:
                pts3D_pred, filename_pred = get3Dskeleton(path, id = i, dep = True, gt=False, csvI=True, pts3Dmet = pts3Dmet, dilt = dilt)
            except:
                pts3D_pred = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
                # missed_count += 1 
                # continue
                filename_pred = str(int(i)).zfill(10)

            try:
                pts3D_gt, filename_gt = get3Dskeleton(path, id = i, dep = True, gt=True, csvI=False, pts3Dmet = pts3Dmet, dilt = dilt)
            except:
                pts3D_gt = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
                # missed_count += 1 
                # continue
                filename_gt = str(int(i)).zfill(10)

            if (filename_pred != filename_gt):
                # print('Warning: File mismatch',filename_gt,filename_pred)
                missed_count += 1
                continue 
            if len(pts3D_gt) == 0 or len(pts3D_pred) == 0 or len(pts3D_pred) < 8:
                missed_count += 1 
                # edist = np.zeros((8,3))
            else:
                pts3D_pred = np.array(pts3D_pred).reshape((8,3))
                pts3D_gt = np.array(pts3D_gt).reshape((8,3))

                KF3D = kf(pts3D_pred, kfinit, kf1D = True, kf2D=True)

            #     if q % 3 == 0:
            #         dff_pos = np.vstack((dff_pos, abs(KF3D-pts3D_pred)))
                
            #     if q % 6 == 0:
            #         dff_vel = np.vstack((dff_vel, abs((KF3D-pts_KF[:,-3:])-(pts3D_pred-pts_full_pred[:,-3:])))

                pts_full_pred = np.hstack((pts_full_pred, pts3D_pred))
                pts_full_gt = np.hstack((pts_full_gt, pts3D_gt))
                pts_KF = np.hstack((pts_KF, KF3D))
            
                if np.any(pts3D_pred[:,2] == 0):
                    holes = holes + np.count_nonzero(pts3D_pred[:,2]==0)
                
                if np.any(pts3D_gt[:,2] == 0):
                    holes_gt = holes_gt + np.count_nonzero(pts3D_pred[:,2]==0)

            if q == 200:
                tot_imgs = len(paths)*q
                break
                
            print('{}/{} in {}/{} := {}, Missed Count: {}'.format(q, len(sampld), n, len(paths) ,path, missed_count), end='\r')
    
    # filter applied: moving Median filter, with fillers
    dfpts_pred_prev = pd.DataFrame(data=np.transpose(pts_full_pred[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])
    dfpts_pred_prev = zerofiller(dfpts_pred_prev)
    dfpts_pred = copy.deepcopy(dfpts_pred_prev)
    dfpts_pred[2::3] = dfpts_pred[2::3].rolling(window=4).median()
    dfpts_pred = interpdf(dfpts_pred)
    dfpts_pred[dfpts_pred.isnull()] = dfpts_pred_prev[dfpts_pred.isnull()]
    
    dfpts_gt_prev = pd.DataFrame(data=np.transpose(pts_full_gt[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])
    dfpts_gt_prev = zerofiller(dfpts_gt_prev)
    dfpts_gt = copy.deepcopy(dfpts_gt_prev)
    dfpts_gt[2::3] = dfpts_gt[2::3].rolling(window=4).median()
    dfpts_gt = interpdf(dfpts_gt)
    dfpts_gt[dfpts_gt.isnull()] = dfpts_gt_prev[dfpts_gt.isnull()]


    dfpts_KF = pd.DataFrame(data=np.transpose(pts_KF[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])


    for col in ['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist']:
        limbs_3D[col] = ed_error(np.array(dfpts_pred[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,2], np.array(dfpts_gt[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,2])
        limbs_2D[col] = ed_error(np.array(dfpts_pred[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,0:2],np.array(dfpts_gt[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,0:2])

        limbs_3D_prev[col] = ed_error(np.array(dfpts_pred_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,2], np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,2])
        limbs_2D_prev[col] = ed_error(np.array(dfpts_pred_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,0:2],np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,0:2])

        limbs_3D_KF[col] = ed_error(np.array(dfpts_KF[col]).reshape((int(dfpts_KF.shape[0]/3),3))[:,2], np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,2])
        limbs_2D_KF[col] = ed_error(np.array(dfpts_KF[col]).reshape((int(dfpts_KF.shape[0]/3),3))[:,0:2],np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,0:2])

    
    # # # # uncomment to visualize skeletons 2D and 3D for imagne no. (imNo)
    # pdb.set_trace()
    imNo =  160; idx = imNo*3
    # vis3Dskeleton(np.array(dfpts_pred_prev[idx:idx+3]).T.reshape((8,3)))
    # vis3Dskeleton(np.array(dfpts_pred[idx:idx+3]).T.reshape((8,3)))
    # vis2Dskeleton_overlay(paths[0], np.array(dfpts_gt[idx:idx+2]).T.reshape((8,2)), np.array(dfpts_pred[idx:idx+2]).T.reshape((8,2)), str(3200+380).zfill(10))
    # vis2Dskeleton_depth(paths[0], np.array(dfpts_pred[idx:idx+2]).T.reshape((8,2)), str(590).zfill(10))
    # vis2Dskeleton_overlay(paths[0], np.array(dfpts_pred[idx:idx+2]).T.reshape((8,2)), np.array(dfpts_KF[idx:idx+2]).T.reshape((8,2)), str(3200+imNo).zfill(10))


    df2D_prev = pd.DataFrame.from_dict(limbs_2D_prev) #euclidean distance without filter 2D
    df3D_prev = pd.DataFrame.from_dict(limbs_3D_prev) #euclidean distance without filter 3D

    df2D = pd.DataFrame.from_dict(limbs_2D) #euclidean distance 2D
    df3D = pd.DataFrame.from_dict(limbs_3D) #euclidean distance 3D

    df2D_KF = pd.DataFrame.from_dict(limbs_2D_KF) #euclidean distance 2D
    df3D_KF = pd.DataFrame.from_dict(limbs_3D_KF) #euclidean distance 3D


    # # fig, [ax1,ax2,ax3] = plt.subplots(1, 3)
    # # fig.set_size_inches(18.5, 10.5)
    # # fig.set_dpi(100)

    # # df3D_prev.plot(kind='box', ax=ax1)
    # # ax1.set(xlabel='Limbs', ylabel='Error [m]')
    # # ax1.tick_params(axis='x', rotation=30)
    # # ax1.set_title('W/o preprocess')
    # # ax1.set_ylim([0,0.30])

    # # df3D.plot(kind='box', ax=ax2)
    # # ax2.set(xlabel='Limbs', ylabel='Error [m]')
    # # ax2.tick_params(axis='x', rotation=30)
    # # ax2.set_title('W MM')
    # # ax2.set_ylim([0,0.30])

    # # df3D_KF.plot(kind='box', ax=ax3)
    # # ax3.set(xlabel='Limbs', ylabel='Error [m]')
    # # ax3.tick_params(axis='x', rotation=30)
    # # ax3.set_title('W KF')
    # # ax3.set_ylim([0,0.30])

    # # fig.suptitle('udp-(z) in meters')
    # # fig.savefig('udp(z).png', dpi=100)

    # # fig, [ax1,ax2,ax3] = plt.subplots(1, 3)
    # # fig.set_size_inches(18.5, 10.5)
    # # fig.set_dpi(100)

    # # df2D_prev.plot(kind='box', ax=ax1)
    # # ax1.set(xlabel='Limbs', ylabel='Error [m]')
    # # ax1.tick_params(axis='x', rotation=30)
    # # ax1.set_title('W/o preprocess')
    # # ax1.set_ylim([0,0.15])

    # # df2D.plot(kind='box', ax=ax2)
    # # ax2.set(xlabel='Limbs', ylabel='Error [m]')
    # # ax2.tick_params(axis='x', rotation=30)
    # # ax2.set_title('W preprocess')
    # # ax2.set_ylim([0,0.15])

    # # df2D_KF.plot(kind='box', ax=ax3)
    # # ax3.set(xlabel='Limbs', ylabel='Error [m]')
    # # ax3.tick_params(axis='x', rotation=30)
    # # ax3.set_title('W KF')
    # # ax3.set_ylim([0,0.15])
    # # fig.suptitle('udp-(x,y) in meters')
    # # fig.savefig('udp(x,y).png', dpi=100)

    # # # temporal_3D_flow(dfpts_pred_prev, dfpts_gt_prev)
    # # temporal_3D_flow(dfpts_pred[2::3], dfpts_gt[2::3], dfpts_KF[2::3], [2,4.5],'z [m]')
    # # temporal_3D_flow(dfpts_pred[1::3], dfpts_gt[1::3], dfpts_KF[1::3], [-0.8,0.8],'y [m]')
    # # temporal_3D_flow(dfpts_pred[0::3], dfpts_gt[0::3], dfpts_KF[0::3], [-0.8,0.8],'x [m]')
    # temporal_3D_flow(dfpts_pred_prev, dfpts_gt_prev)
    # temporal_3D_flow(dfpts_pred[2::3], dfpts_gt[2::3])
    # temporal_3D_flow(pts_full_pred[:,1:], pts_full_gt[:,1:])
    # plt.savefig('udp_analysis.png')
    # fig.suptitle('udp performance')
    # fig, axes = plt.subplots(1, 2)
    # sns.boxplot(ax=axes[0, 0], data=df2D)
    # sns.boxplot(ax=axes[0, 1], data=df3D)
    # sns_plot = sns.histplot(df2D, x = 'pixels')
    # sns_plot.savefig("output_udp.png")
    # axes[0,0].xlabel('Limbs');axes[0,0].ylabel('Error [pixels]')
    # axes[0,1].xlabel('Limbs');axes[0,1].ylabel('Error [m]')
    # paired_ttest(dfpts_pred, dfpts_gt)
    # # # # sm.qqplot(dfpts_pred['Right-Wrist'][2::3], line='45')

    # print('\n ===========  GT vs Pred ============= \n')
    # print('\n KL-Divergence')
    # kl_divergence(dfpts_pred, dfpts_gt)
    # print('\n Paired T-test')
    # paired_ttest(dfpts_pred, dfpts_gt)

    # print('\n =========== Pred vs Filtered Pred ============= \n')
    # print('\n KL-Divergence')
    # kl_divergence(dfpts_pred, dfpts_pred_prev)
    # print('\n Paired T-test \n')
    # paired_ttest(dfpts_pred, dfpts_pred_prev)
    
    # print('\n =========== GT and Pred SNR ============= \n')
    # print('\nSNR GT: \n', np.mean(dfpts_gt[2::3]) / np.abs(np.mean(dfpts_gt_prev[2::3] - dfpts_gt[2::3])))
    # print('\nSNR Pred: \n', np.mean(dfpts_pred[2::3]) / np.abs(np.mean(dfpts_pred_prev[2::3]-dfpts_pred[2::3])))
    # print('\n ============================================ \n')

    # sns.pairplot(dfpts_pred-dfpts_gt)
    # plt.savefig('pairplot.png')
    # sns.pairplot(dfpts_pred-dfpts_gt, kind="kde")
    # plt.savefig('pairplot_kde.png')
    outlier_elements3D_prev, outlier_imageSample3D_prev = outliers_count(df3D_prev)
    outlier_elements3D, outlier_imageSample3D = outliers_count(df3D)
    outlier_elements3D_KF, outlier_imageSample3D_KF = outliers_count(df3D_KF)
    plt.show()
    print('\n','missed (%):', missed_count/tots*100, '\n 3D outlier element Prev (%):', outlier_elements3D_prev, '\n images with 3D outliers Prev (%):', outlier_imageSample3D_prev, '\n 3D outlier element MM (%):', outlier_elements3D, '\n images with 3D outliers MM (%):', outlier_imageSample3D, '\n 3D outliers element KF (%):', outlier_elements3D_KF, '\n images with 3D outliers KF (%):', outlier_imageSample3D_KF)
    # dff_pos2 = np.reshape(dff_pos,(int(dff_pos.shape[0]/8),8,3)); dff_vel2 = np.reshape(dff_vel,(int(dff_vel.shape[0]/8),8,3))
    # print(np.std(dff_pos2,axis=0), np.mean(np.std(dff_pos2,axis=0),axis=0),'\n' np.std(dff_vel2,axis=0), np.mean(np.std(dff_vel2,axis=0),axis=0))

    # pdb.set_trace()
    return eds2D, eds3D

def compareGT(paths):
    eds2D = np.empty([8,1]); eds3D = np.empty([8,1]); missed_count=0;n=0;q = 0;outliers_count3D=0;outliers_count2D=0;tots=0;failed_count_2D=0;failed_count_3D=0
    pts_full_pred = np.empty([8,1]); pts_full_gt = np.empty([8,1]);pts_KF = np.empty([8,1]); limbs_2D = {}; limbs_3D = {};limbs_3D_prev={};limbs_2D_prev={};
    limbs_3D_KF = {}; limbs_2D_KF = {}; std_acc=500; x_std_meas=12.7; y_std_meas=17.4;std_acc_1D=500; x_std_1D=0.15;

    rgb0 = os.path.join(paths[0],'rgb');rgb1 = os.path.join(paths[1],'rgb')
    rgbImgs0 = os.listdir(rgb0);rgbImgs1 = os.listdir(rgb1)

    KF_head = KalmanFilter(1/15, 1, 1, std_acc, 16.41, 29.49)
    KF1D_head = KalmanFilter1D(1/15, 1, std_acc_1D, 0.10)
    KF_midS = KalmanFilter(1/15, 1, 1, std_acc, 11.40, 19.86)
    KF1D_midS = KalmanFilter1D(1/15, 1, std_acc_1D, 0.10)
    KF_RS = KalmanFilter(1/15, 1, 1, std_acc, 12.63, 20.64)
    KF1D_RS = KalmanFilter1D(1/15, 1, std_acc_1D, 0.10)
    KF_RE = KalmanFilter(1/15, 1, 1, std_acc, 15.15, 15.57)
    KF1D_RE = KalmanFilter1D(1/15, 1, std_acc_1D, 0.08)
    KF_RW = KalmanFilter(1/15, 1, 1, std_acc, 14.58, 10.53)
    KF1D_RW = KalmanFilter1D(1/15, 1, std_acc_1D, 0.09)
    KF_LS = KalmanFilter(1/15, 1, 1, std_acc, 11.51, 19.60)
    KF1D_LS = KalmanFilter1D(1/15, 1, std_acc_1D, 0.45)
    KF_LE = KalmanFilter(1/15, 1, 1, std_acc, 10.60, 11.81)
    KF1D_LE = KalmanFilter1D(1/15, 1, std_acc_1D, 0.15)
    KF_LW = KalmanFilter(1/15, 1, 1, std_acc, 9.70, 12.19)
    KF1D_LW = KalmanFilter1D(1/15, 1, std_acc_1D, 0.08)

    diff_imgs = 0
    # sampld = np.random.randint(0+diff_imgs, int(len(rgbImgs))-diff_imgs,  size=int(len(rgbImgs)/(3))).tolist() #sampling 33% of ids from a folder of data randomly
    sampld = range(0,900)
    # sampld = np.random.randint(0+diff_imgs, int(len(rgbImgs))-diff_imgs,  size=int(len(rgbImgs))).tolist() #sampling 33% of ids from a folder of data randomly
    tots = tots + len(sampld)
    for i in sampld:
        q = q + 1;here=0
        # img = cv2.imread(rgb+imgNames)
        try:
            pts3D_pred, filename_pred = get3Dskeleton(paths[1], id = i, dep = True, gt=True, csvI=False)
        except:
            pts3D_pred = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
            # missed_count += 1 
            # continue
            filename_pred = str(int(i)).zfill(10)

        try:
            pts3D_gt, filename_gt = get3Dskeleton(paths[0], id = i, dep = True, gt=True, csvI=False)
        except:
            pts3D_gt = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
            # missed_count += 1 
            # continue
            filename_gt = str(int(i)).zfill(10)

        if (filename_pred != filename_gt):
            # print('Warning: File mismatch',filename_gt,filename_pred)
            missed_count += 1
            continue 
        if len(pts3D_gt) == 0 or len(pts3D_pred) == 0 or len(pts3D_pred) < 8:
            missed_count += 1 
            # edist = np.zeros((8,3))
        else:
            pts3D_pred = np.array(pts3D_pred).reshape((8,3))
            pts3D_gt = np.array(pts3D_gt).reshape((8,3))

            x_head = KF_head.predict()
            x1_head = KF_head.update((pts3D_pred[0,0], pts3D_pred[0,1]))
            x1D_head = KF1D_head.predict() 
            x11D_head = KF1D_head.update(pts3D_pred[0,2])

            x_midS = KF_midS.predict()
            x1_midS = KF_midS.update((pts3D_pred[1,0], pts3D_pred[1,1]))
            x1D_midS = KF1D_midS.predict() 
            x11D_midS = KF1D_midS.update(pts3D_pred[1,2])

            x_RS = KF_RS.predict()
            x1_RS = KF_RS.update((pts3D_pred[2,0], pts3D_pred[2,1]))
            x1D_RS = KF1D_RS.predict() 
            x11D_RS = KF1D_RS.update(pts3D_pred[2,2])

            x_RE = KF_RE.predict()
            x1_RE = KF_RE.update((pts3D_pred[3,0], pts3D_pred[3,1]))
            x1D_RE = KF1D_RE.predict() 
            x11D_RE = KF1D_RE.update(pts3D_pred[3,2])

            x_RW = KF_RW.predict()
            x1_RW = KF_RW.update((pts3D_pred[4,0], pts3D_pred[4,1]))
            x1D_RW = KF1D_RW.predict() 
            x11D_RW = KF1D_RW.update(pts3D_pred[4,2])

            x_LS = KF_LS.predict()
            x1_LS = KF_LS.update((pts3D_pred[5,0], pts3D_pred[5,1]))
            x1D_LS = KF1D_LS.predict() 
            x11D_LS = KF1D_LS.update(pts3D_pred[5,2])

            x_LE = KF_LE.predict()
            x1_LE = KF_LE.update((pts3D_pred[6,0], pts3D_pred[6,1]))
            x1D_LE = KF1D_LE.predict() 
            x11D_LE = KF1D_LE.update(pts3D_pred[6,2])

            x_LW = KF_LW.predict()
            x1_LW = KF_LW.update((pts3D_pred[7,0], pts3D_pred[7,1]))
            x1D_LW = KF1D_LW.predict() 
            x11D_LW = KF1D_LW.update(pts3D_pred[7,2]) 


            KF3D = np.array(
            [
                    [x1_head[0,0], x1_head[0,1],x11D_head[0,0]],
                    [x1_midS[0,0], x1_midS[0,1],x11D_midS[0,0]],
                    [x1_RS[0,0], x1_RS[0,1],x11D_RS[0,0]],
                    [x1_RE[0,0], x1_RE[0,1],x11D_RE[0,0]],
                    [x1_RW[0,0], x1_RW[0,1],x11D_RW[0,0]],
                    [x1_LS[0,0], x1_LS[0,1],x11D_LS[0,0]],
                    [x1_LE[0,0], x1_LE[0,1],x11D_LE[0,0]],
                    [x1_LW[0,0], x1_LW[0,1],x11D_LW[0,0]],
            ])

            pts_full_pred = np.hstack((pts_full_pred, pts3D_pred))
            pts_full_gt = np.hstack((pts_full_gt, pts3D_gt))
            pts_KF = np.hstack((pts_KF, KF3D))

        # if q == 200:
        #     break
            
        print('{}/{}, Missed Count: {}'.format(q, len(sampld), missed_count), end='\r')
    
    # filter applied: moving Median filter, with fillers
    dfpts_pred_prev = pd.DataFrame(data=np.transpose(pts_full_pred[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])
    dfpts_pred_prev = zerofiller(dfpts_pred_prev)
    dfpts_pred = copy.deepcopy(dfpts_pred_prev)
    dfpts_pred[2::3] = dfpts_pred[2::3].rolling(window=4).median()
    dfpts_pred[dfpts_pred.isnull()] = dfpts_pred_prev[dfpts_pred.isnull()]
    
    dfpts_gt_prev = pd.DataFrame(data=np.transpose(pts_full_gt[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])
    dfpts_gt_prev = zerofiller(dfpts_gt_prev)
    dfpts_gt = copy.deepcopy(dfpts_gt_prev)
    dfpts_gt[2::3] = dfpts_gt[2::3].rolling(window=3).median()
    dfpts_gt[dfpts_gt.isnull()] = dfpts_gt_prev[dfpts_gt.isnull()]


    dfpts_KF = pd.DataFrame(data=np.transpose(pts_KF[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])

    for col in ['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist']:
        limbs_3D[col] = ed_error(np.array(dfpts_pred[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,2], np.array(dfpts_gt[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,2])
        limbs_2D[col] = ed_error(np.array(dfpts_pred[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,0:2],np.array(dfpts_gt[col]).reshape((int(dfpts_pred.shape[0]/3),3))[:,0:2])

        limbs_3D_prev[col] = ed_error(np.array(dfpts_pred_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,2], np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,2])
        limbs_2D_prev[col] = ed_error(np.array(dfpts_pred_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,0:2],np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,0:2])

        limbs_3D_KF[col] = ed_error(np.array(dfpts_KF[col]).reshape((int(dfpts_KF.shape[0]/3),3))[:,2], np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,2])
        limbs_2D_KF[col] = ed_error(np.array(dfpts_KF[col]).reshape((int(dfpts_KF.shape[0]/3),3))[:,0:2],np.array(dfpts_gt_prev[col]).reshape((int(dfpts_pred_prev.shape[0]/3),3))[:,0:2])



    df2D_prev = pd.DataFrame.from_dict(limbs_2D_prev) #euclidean distance without filter 2D
    df3D_prev = pd.DataFrame.from_dict(limbs_3D_prev) #euclidean distance without filter 3D

    df2D = pd.DataFrame.from_dict(limbs_2D) #euclidean distance 2D
    df3D = pd.DataFrame.from_dict(limbs_3D) #euclidean distance 3D

    df2D_KF = pd.DataFrame.from_dict(limbs_2D_KF) #euclidean distance 2D
    df3D_KF = pd.DataFrame.from_dict(limbs_3D_KF) #euclidean distance 3D

    # imNo =  df2D.loc[(df2D>10).any(axis=1)].index[-15]
    # idx = imNo*3
    # vis2Dskeleton_overlay(paths[0], np.array(dfpts_gt[idx:idx+2]).T.reshape((8,2)), np.array(dfpts_pred[idx:idx+2]).T.reshape((8,2)), str(imNo).zfill(10))

    # fig, [ax1,ax2,ax3] = plt.subplots(1, 3)

    # df3D_prev.plot(kind='box', ax=ax1)
    # ax1.set(xlabel='Limbs', ylabel='Error [m]')
    # ax1.tick_params(axis='x', rotation=30)
    # ax1.set_title('W/o preprocess')
    # ax1.set_ylim([0,0.15])

    # df3D.plot(kind='box', ax=ax2)
    # ax2.set(xlabel='Limbs', ylabel='Error [m]')
    # ax2.tick_params(axis='x', rotation=30)
    # ax2.set_title('W MM')
    # ax2.set_ylim([0,0.15])

    # df3D_KF.plot(kind='box', ax=ax3)
    # ax3.set(xlabel='Limbs', ylabel='Error [m]')
    # ax3.tick_params(axis='x', rotation=30)
    # ax3.set_title('W KF')
    # ax3.set_ylim([0,0.15])

    # fig.suptitle('Manual GTs')

    # fig, [ax1,ax2,ax3] = plt.subplots(1, 3)

    # df2D_prev.plot(kind='box', ax=ax1)
    # ax1.set(xlabel='Limbs', ylabel='Error [pixels]')
    # ax1.tick_params(axis='x', rotation=30)
    # ax1.set_title('W/o preprocess')
    # ax1.set_ylim([0,15])

    # df2D.plot(kind='box', ax=ax2)
    # ax2.set(xlabel='Limbs', ylabel='Error [pixels]')
    # ax2.tick_params(axis='x', rotation=30)
    # ax2.set_title('W preprocess')
    # ax2.set_ylim([0,15])

    # df2D_KF.plot(kind='box', ax=ax3)
    # ax3.set(xlabel='Limbs', ylabel='Error [pixels]')
    # ax3.tick_params(axis='x', rotation=30)
    # ax3.set_title('W KF')
    # ax3.set_ylim([0,15])

    # fig.suptitle('Manual GTs')
    # # temporal_3D_flow(dfpts_pred_prev, dfpts_gt_prev)
    # temporal_3D_flow(dfpts_pred, dfpts_gt, dfpts_KF)
    # temporal_3D_flow(pts_full_pred[:,1:], pts_full_gt[:,1:])
    # plt.savefig('udp_analysis.png')
    # fig.suptitle('udp performance')
    # fig, axes = plt.subplots(1, 2)
    # sns.boxplot(ax=axes[0, 0], data=df2D)
    # sns.boxplot(ax=axes[0, 1], data=df3D)
    # sns_plot = sns.histplot(df2D, x = 'pixels')
    # sns_plot.savefig("output_udp.png")
    # axes[0,0].xlabel('Limbs');axes[0,0].ylabel('Error [pixels]')
    # axes[0,1].xlabel('Limbs');axes[0,1].ylabel('Error [m]')
    # paired_ttest(dfpts_pred, dfpts_gt)
    sm.qqplot(dfpts_pred['Right-Wrist'][2::3], line='45')

    print('\n =========== MGT Seq1 vs Seq2 ============= \n')
    print('\n KL-Divergence')
    kl_divergence(dfpts_pred, dfpts_gt)
    print('\n Paired T-test')
    paired_ttest(dfpts_pred, dfpts_gt)

    print('\n =========== MGT Seq2 vs FilteredSeq ============= \n')
    print('\n KL-Divergence')
    kl_divergence(dfpts_pred, dfpts_pred_prev)
    print('\n Paired T-test \n')
    paired_ttest(dfpts_pred, dfpts_pred_prev)
    
    print('\n =========== MGT SNR ============= \n')
    print('\nSNR Seq1: \n', np.mean(dfpts_gt[2::3]) / np.abs(np.mean(dfpts_gt_prev[2::3] - dfpts_gt[2::3])))
    print('\nSNR Seq2: \n', np.mean(dfpts_pred[2::3]) / np.abs(np.mean(dfpts_pred_prev[2::3]-dfpts_pred[2::3])))

    outlier_elements2D, outlier_imageSample2D = outliers_count(df2D)
    outlier_elements3D, outlier_imageSample3D = outliers_count(df3D)
    outlier_elements3D_KF, outlier_imageSample3D_KF = outliers_count(df3D_KF)
    print('\n','missed (%):', missed_count/tots*100, '\n 2D outlier element MM (%):', outlier_elements2D, '\n 3D outlier element MM (%):', outlier_elements3D, '\n images with 2D outliers MM (%):', outlier_imageSample2D, '\n images with 3D outliers MM (%):', outlier_imageSample3D, '\n 3D outliers element KF (%):', outlier_elements3D_KF, '\n images with 3D outliers KF (%):', outlier_imageSample3D_KF)
    plt.show()

def compCSV(paths):
    missed_count=0;n=0;q = 0;outliers_count3D=0;outliers_count2D=0;tots=0;failed_count_2D=0;failed_count_3D=0
    pts_full_pred = np.empty([8,1]); pts_full_gt = np.empty([8,1]); 

    rgb0 = os.path.join(paths[0],'rgb');
    rgbImgs = os.listdir(rgb0);

    diff_imgs = 0
    sampld = range(0,len(rgbImgs))
    tots = tots + len(sampld)
    for i in sampld:
        q = q + 1;here=0
        try:
            pts3D_pred, filename_pred = get3Dskeleton(paths[0], id = i, dep = True, gt=True, csvI=False, csvName = '20210822_184335_preds_darkpose.csv')
        except:
            pts3D_pred = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
            filename_pred = str(int(i)).zfill(10)

        try:
            pts3D_gt, filename_gt = get3Dskeleton(paths[0], id = i, dep = True, gt=True, csvI=False, csvName = '20210822_184335_preds_HigherHRNet.csv')
        except:
            pts3D_gt = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
            filename_gt = str(int(i)).zfill(10)

        if (filename_pred != filename_gt):
            # print('Warning: File mismatch',filename_gt,filename_pred)
            missed_count += 1
            continue 
        if len(pts3D_gt) == 0 or len(pts3D_pred) == 0 or len(pts3D_pred) < 8:
            missed_count += 1 
            # edist = np.zeros((8,3))
        else:
            pts3D_pred = np.array(pts3D_pred).reshape((8,3))
            pts3D_gt = np.array(pts3D_gt).reshape((8,3))

            pts_full_pred = np.hstack((pts_full_pred, pts3D_pred))
            pts_full_gt = np.hstack((pts_full_gt, pts3D_gt))


        print('{}/{}, Missed Count: {}'.format(q, len(sampld), missed_count), end='\r')
    
    # filter applied: moving Median filter, with fillers
    dfpts_pred_prev = pd.DataFrame(data=np.transpose(pts_full_pred[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])
    dfpts_pred_prev = zerofiller(dfpts_pred_prev)
    dfpts_pred = copy.deepcopy(dfpts_pred_prev)
    # dfpts_pred[2::3] = dfpts_pred[2::3].rolling(window=4).median()
    dfpts_pred[dfpts_pred.isnull()] = dfpts_pred_prev[dfpts_pred.isnull()]
    
    dfpts_gt_prev = pd.DataFrame(data=np.transpose(pts_full_gt[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])
    dfpts_gt_prev = zerofiller(dfpts_gt_prev)
    dfpts_gt = copy.deepcopy(dfpts_gt_prev)
    # dfpts_gt[2::3] = dfpts_gt[2::3].rolling(window=3).median()
    dfpts_gt[dfpts_gt.isnull()] = dfpts_gt_prev[dfpts_gt.isnull()]

    # fig, [ax1,ax2,ax3] = plt.subplots(1,3)
    # sns.boxplot(data=dfpts_gt[0::3], ax=ax1)
    # sns.boxplot(data=dfpts_gt[1::3], ax=ax2)
    # sns.boxplot(data=dfpts_gt[2::3], ax=ax3)

    # fig,[ax1,ax2] = plt.subplots(1,2)
    # boxplot2D(dfpts_gt, ax=ax1)
    # sns.boxplot(data=dfpts_gt[2::3], ax=ax2)
    # sns.displot(data=dfpts_gt[0::3], kind="kde")
    # # def func3(x, y):
    # #     return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2 + y**2))

    # # fig = plt.figure(frameon=False)
    # # imNo =  42; idx = imNo*3
    # # img = cv2.imread(paths[0] + '/rgb/' +  str(0).zfill(10) + '.png')
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # im1 = plt.imshow(img,cmap=plt.cm.gray, alpha=0.5)
    # # x = dfpts_gt[0::3].values.reshape(-1)
    # # y = dfpts_gt[1::3].values.reshape(-1)

    # # heatmap = np.zeros((img.shape[0],img.shape[1]))
    # # for i in range(len(x)):
    # #     heatmap[int(y[i]),int(x[i])] = heatmap[int(y[i]),int(x[i])]  + 1
    # # im2 = plt.imshow(heatmap, plt.cm.viridis,alpha=0.5, interpolation='bilinear')
    # im2 = plt.imshow(Z2,cmap=plt.cm.viridis,interpolation='bilinear')
    # ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45);ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45);ax3.set_xticklabels(ax3.get_xticklabels(),rotation=45)
    # ax1.set_ylabel('x [pixels]');ax2.set_ylabel('y [pixels]');ax3.set_ylabel('z [m]');
    plt.show()



# paths = ['E:\Eatsense2/20210623_150016', 'E:\Eatsense2/20210509_233500', 'E:/EatSense2/20210505_234729',]
# paths = ['E:\Eatsense2/20210623_150016',]

# paths = ['F:\Ahmed/20210505_234729', 'F:\Ahmed/20210509_233500', 'F:\Ahmed/20210623_150016']
# paths = ['D:\PhD Edinburgh\Codes\labellers\labeller/randImgs']
# kpv = []
# lst_dir = os.listdir(paths[0]+'/rgb/')
# openpose_missed = [
# 29 ,
# 551 ,
# 595 ,
# 605 ,
# 879 ,
# 991 ,
# 1223 ,
# 1243 ,
# 1253 ,
# 1554 ,
# 2000 ,
# 2253 ,
# 2269 ,
# 2909 ,
# 3164 ,
# 3311 ,
# 3560 ,
# 4208 ,
# 4534 ,
# 4738 ,
# 5344 ,
# 5935 ,
# 6174 ,
# 6460 ,
# 6561 ,
# 6952 ,
# 7206 ,
# 8186 ,
# 8651 ,
# 8667 ,
# 9064 ,
# 11028 ,
# 11679 ]
# pts_full_pred = np.empty([8,1]); pts_full_gt = np.empty([8,1]);
# for img in lst_dir:
#     idx = int(img.split('.')[0])
#     # if idx in openpose_missed:
#     #     continue
#     try:
#         # pts2D_gt, filename = get2Dskeleton(path=paths[0], id = idx, dep = True, gt=True, csvI=False, csvName= 'randImgs.csv')
#         pts3D_gt, filename_gt = get3Dskeleton(paths[0], id = idx, dep = True, gt=True, csvI=False, csvName = 'randImgs.csv')
#     except:
#         print('gt',idx)
#         continue
#     try:
#         # pts2D_preds, filename = get2Dskeleton(path=paths[0], id = idx, dep = True, gt=True, csvI=False, csvName= 'outfiles\csvs/_preds_darkpose.csv')
#         pts3D_pred, filename_pred = get3Dskeleton(paths[0], id = idx, dep = True, gt=True, csvI=False, csvName = 'outfiles\csvs/_preds_darkpose.csv')
#     except:
#         print('preds',idx)
#         continue

#     pts3D_pred = np.array(pts3D_pred).reshape((8,3))
#     pts3D_gt = np.array(pts3D_gt).reshape((8,3))

#     pts_full_pred = np.hstack((pts_full_pred, pts3D_pred))
#     pts_full_gt = np.hstack((pts_full_gt, pts3D_gt))

#     dfpts_pred = pd.DataFrame(data=np.transpose(pts_full_pred[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])
#     dfpts_gt = pd.DataFrame(data=np.transpose(pts_full_gt[:,1:]), columns=['head','Mid-Shoulder','Right-Shoulder','Right-Elbow','Right-Wrist','Left-shoulder','Left-Elbow','Left-Wrist'])

# pdb.set_trace()

# thres = 2.4
# dt = dfpts_pred.to_numpy().reshape(-1,1)
# gt= dfpts_gt.to_numpy().reshape(-1,1)

# dq = (dt - gt) ** 2
# idd = dq < thres**2
# pq = dq[idd]

# ed = np.sum(dq[idd])/len(dq[idd])
# print(ed)
# wrong_preds = (len(dq) - len(pq))/len(dq)
# print(wrong_preds)

# print(ed_error2(dfpts_pred.to_numpy(), dfpts_gt.to_numpy()))
# print(mean_squared_error(dfpts_pred, dfpts_gt))

# dx = (dt[0::2] - gt[0::2])**2
# idx = dx < thres**2
# dy = (dt[1::2] - gt[1::2])**2
# idy = dy < thres**2
# dz = (dt[1::2] - gt[1::2])**2
# idz = dz < thres**2
# px = dx[idx & idy & idz]; py = dy[idx & idy & idz]; pz = dy[idx & idy & idz]; 
# ed = np.sum(np.hstack((px,py)))/len(np.hstack((px,py)))

# generate_preds(path)
# paths = ['D:\PhD Edinburgh\RealSenseTests\EatSense/20210518_230219']
# eds2D, eds3D = VisualizeError(paths)

# paths = ['D:\PhD Edinburgh\RealSenseTests\EatSense\Test/Test1', 'D:\PhD Edinburgh\RealSenseTests\EatSense\Test/Test2']
# compareGT(paths)

# paths = ['D:\PhD Edinburgh\RealSenseTests\EatSense/20210822_184335']
# compCSV(paths)
