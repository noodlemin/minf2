import zipfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os, pdb, cv2, copy, datetime, csv, re
import pandas as pd
# from udp.body import Body
import seaborn as sns
from scipy.ndimage import gaussian_filter, sobel
from scipy import stats
from scipy.interpolate import CubicSpline, Rbf, interp2d
from scipy.special import rel_entr, kl_div
# import pyrealsense2 as rs

# from KalmanFilter import KalmanFilter, KalmanFilter1D
# import statsmodels.api as sm
from math import log2, log

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from errorVis import proj2Dto3D, get2Dskeleton, get3Dskeleton, interpdf, make_intrinsics, kfInit, kf, temporal_3D_flow

fps_dict = {'20210518_230219':15, '20210523_202300':15, '20210529_153708':30, '20210530_153343':30, '20210609_154241':30}

def zerofiller(df):
    cols = ['head_x','head_y','head_z','Mid-Shoulder_x','Mid-Shoulder_y','Mid-Shoulder_z','Right-Shoulder_x','Right-Shoulder_y','Right-Shoulder_z', 
    'Right-Elbow_x','Right-Elbow_y','Right-Elbow_z','Right-Wrist_x','Right-Wrist_y','Right-Wrist_z','Left-shoulder_x','Left-shoulder_y','Left-shoulder_z',
    'Left-Elbow_x','Left-Elbow_y','Left-Elbow_z','Left-Wrist_x','Left-Wrist_y','Left-Wrist_z']
    df_temp = copy.deepcopy(df)
    df_temp2 = df[cols]
    dfX = df_temp2[df_temp2.columns[0::3]]
    dfY = df_temp2[df_temp2.columns[1::3]]
    dfZ = df_temp2[df_temp2.columns[2::3]]
    idx = dfX.index[(dfX == 0).all(axis=1)]
    idy = dfY.index[(dfY == 0).all(axis=1)]
    idz = dfZ.index[(dfZ == 0).all(axis=1)]
    window = 1

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

    df_temp2[df_temp2.columns[0::3]] = dfX
    df_temp2[df_temp2.columns[1::3]] = dfY
    df_temp2[df_temp2.columns[2::3]] = dfZ


    df_temp[cols] = df_temp2[cols]
    return df_temp

def filt(action, project=''):
    projects_both = ['20210518_230219']; projects_one=[]
    if 'snippet' in action:
        action = 'crop action'
    if action == 'pick food from utensil with aboth hands':
        action = 'pick food from utensil with both hands'
    if action == 'pick up - cup/glass' or action == 'pick up a glass/cup':
        action = 'pick up a cup/glass'
    if action == 'pick up no tools':
        action = 'pick up no tool'
    if action == 'end activity - stand up' or action == 'finished eating - stand up':
        action = 'finish food - end activity'
    if action == 'pick food from utensil with tools in both   hands':
        action = 'pick food from utensil with tools in both hands'
    if action == 'pick food from utensil with a tool' or action == 'pick food from utensil with tool':
        action = 'pick food from utensil with tool in one hand'
    if action == 'pick food from utensil with tools':
        action = 'pick food from utensil with tools in both hands'
    if action == 'put the tool back':
        action = 'put one tool back'
    if action == 'put both the tools back':
        action = 'put both tools back'
    if action == 'clean hands/mouth':
        action = 'clean mouth/hands'
    if action == 'pick up tool with one hand' or action == 'pick up tools in one hand' or action == 'pick up tool in one tool':
        action = 'pick up a tool with one hand'
    if action == 'pick up tools in both hands' or action == 'pick up a tool with both hands':
        action = 'pick up tools with both hands'
    if action == 'pick food from utensil with tool by one hand' or action == 'pick food from utensil with tools in one hand':
        action = 'pick food from utensil with tool in one hand'
    if action == 'pick food from the utensil with no tool':
        action = 'pick food from utensil with one hand'
    if (action == 'pick food from utensil' and project in projects_both):
        action= 'pick food from utensil with tools in both hands'
    if (action == 'pick food from utensil' and project in projects_one):
        action= 'pick food from utensil with one hand'
    return action

def getActions(path, project):
    files = os.listdir(path)
    for f in files:
        if project in f:
            actions = []; times=[];projects=[];imgs=[];it=0
            if f.split('.')[-1] == 'csv' and f.split('_')[-1] != 'HL.csv':
                df_temp1 = pd.read_csv(path+'/'+f,header=None, skiprows=2)
                df_temp1.columns=['gibberish','video_file','start','end','action_dict']
                video_name = re.search(r"\[([A-Za-z0-9_]+)\]", eval(df_temp1['video_file'][0])[0]).group(1)
                for index,row in df_temp1.iterrows():
                    action = eval(row['action_dict'])[video_name].lower()
                    action = filt(action, video_name)
                    if video_name in fps_dict:
                        fps = fps_dict[video_name]
                    else:
                        fps = 30
                    for it in range(round(row['start']*fps), round(row['end']*fps)):
                        actions.append(action)
                        projects.append(video_name)
                        imgs.append(it)
                df_new = pd.DataFrame(list(zip(projects,actions,imgs)),columns =['Project', 'Action','Imgs'])
            elif f.split('_')[-1] == 'HL.csv':
                df_temp1 = pd.read_csv(path+'/'+f)
                lst = []; act_count_frames=0
                df_temp1.sort_values(by=['imgName'])
                act0 = 'Sitting Down - Start activity'.lower()
                for index,row in df_temp1.iterrows():
                    if pd.isna(row['path']):
                        continue
                    project = str(row['path']).split('\\')[-2]
                    action = row['Action'].lower(); 
                    action = filt(action, project)
                    actions.append(action)
                    projects.append(project)
                    imgs.append(int(row['imgName']))
                df_new = pd.DataFrame(list(zip(projects,actions,imgs)),columns =['Project', 'Action','Imgs'])
    return df_new

def getPoses(path, project):
    n=0;q = 0;pts_full_pred = np.empty([8*3,1]); dilt = False; pts3Dmet = True; holes=0;imgs=[];imgname = {};missed_count=0
    cols = ['head_x','head_y','head_z','Mid-Shoulder_x','Mid-Shoulder_y','Mid-Shoulder_z','Right-Shoulder_x','Right-Shoulder_y','Right-Shoulder_z', 
    'Right-Elbow_x','Right-Elbow_y','Right-Elbow_z','Right-Wrist_x','Right-Wrist_y','Right-Wrist_z','Left-shoulder_x','Left-shoulder_y','Left-shoulder_z',
    'Left-Elbow_x','Left-Elbow_y','Left-Elbow_z','Left-Wrist_x','Left-Wrist_y','Left-Wrist_z']
    files = os.listdir(path)
    # print(files)
    for f in files:
        if project in f:
            if '.zip' not in f:
                print(path+f)
                # with zipfile.ZipFile(path+f, mode="r") as a:
                #     a.
                    # subfiles = list(filter(lambda a: a.startswith("subdir"), archive.namelist()))
                    # for i in archive.namelist():
                    #     if i.startswith("subdir"):
                    #         print(i)
                    #     print(i)
                subfiles = os.listdir(path +'/' + f)
                    # print(subfiles)
                for subfile in subfiles:
                    if ('_preds_HigherHRNet' in subfile) or subfile == project or ((project == '20210518_230219') and ('_preds_udp' in subfile)) or (subfile == '20210705_225631.csv'):
                        csvfile = path +'/'+ f + '/' + subfile
                        rgb = os.path.join(path +'/' + f,'rgb'); dep = os.path.join(path +'/' + f,'depth')
                        rgbImgs = os.listdir(rgb)
                        sampld = len(rgbImgs)
                        df = pd.read_csv(csvfile)
                        # df = df.sort_values(by=['imgName'])
                        df.sort_values(['imgName'], inplace=True)
                        for i in range(sampld):
                            print(f'{i}/{sampld}', end='\r')
                            try:
                                row = df.loc[df['imgName'].values == i]
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
                                depImg = np.load(dep+'/'+filename+'.npy')
                                imgs.append(int(filename))
                                pts3D_pred = proj2Dto3D(pts2D, depImg, pts3Dmet = pts3Dmet, dilt = dilt)
                            except:
                                pts3D_pred = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
                                imgs.append(i)

                            if len(pts3D_pred) == 0 or len(pts3D_pred) < 8:
                                missed_count += 1 
                            else:
                                pts3D_pred = np.array(pts3D_pred).reshape((8*3,1))
                                pts_full_pred = np.hstack((pts_full_pred, pts3D_pred))
                        
                            # if i == 300:
                            #     break

                            if pts_full_pred[:,1:].shape[1] != len(imgs):
                                print('Length of points and imgs is not the same')
                                print(pts_full_pred[:,1:].shape[1], len(imgs))
                                imgs.pop()

                        
                        pdb.set_trace()
                    dfpts_pred_prev = pd.DataFrame(data=np.transpose(pts_full_pred[:,1:]), columns=cols)
                    dfpts_pred_prev = zerofiller(dfpts_pred_prev)
                    dfpts_pred = copy.deepcopy(dfpts_pred_prev)
                    dfpts_pred[dfpts_pred.columns[2::3]] = dfpts_pred[dfpts_pred.columns[2::3]].rolling(window=4).median()
                    # dfpts_pred = interpdf(dfpts_pred)
                    dfpts_pred[dfpts_pred.isnull()] = dfpts_pred_prev[dfpts_pred.isnull()]
                    # temporal_3D_flow(dfpts_pred[dfpts_pred.columns[2::3]], dfpts_pred[dfpts_pred.columns[2::3]], df3 = None, lim=[0,4.5], strng='Title')
    dfpts_pred['Imgs'] = imgs
    return dfpts_pred

def getVelocities(df):
    cols = ['head_velx','head_vely','head_velz','Mid-Shoulder_velx','Mid-Shoulder_vely','Mid-Shoulder_velz','Right-Shoulder_velx','Right-Shoulder_vely','Right-Shoulder_velz',
    'Right-Elbow_velx','Right-Elbow_vely','Right-Elbow_velz','Right-Wrist_velx','Right-Wrist_vely','Right-Wrist_velz','Left-shoulder_velx','Left-shoulder_vely','Left-shoulder_velz',
    'Left-Elbow_velx','Left-Elbow_vely','Left-Elbow_velz','Left-Wrist_velx','Left-Wrist_vely','Left-Wrist_velz', 'Imgs']
    df_temp = copy.deepcopy(df)
    df_temp.columns = cols
    df_temp[df_temp.columns[:-1]] = df[df.columns[:-1]].diff()
    df_temp = df_temp.fillna(0)
    return df_temp

def getEverything(path, action_path, exportPath):
    # files = os.listdir(path); 
    files = ['20210529_150552', '20210529_150552.zip','20210609_221133', '20210609_221133.zip']
    qq = 0
    for f in files:
        if '.zip' not in f:
            print(f'{qq}/{int(len(files)/2)} ==> {f}')
            df_actions = getActions(action_path, f)
            df_poses = getPoses(path, f)
            df_vel = getVelocities(df_poses)
            df_temp = pd.merge(df_actions, df_poses, how='outer', on='Imgs')
            df = pd.merge(df_temp, df_vel, how='outer', on='Imgs')
            df = df.dropna()
            df.sort_values(['Imgs'],inplace=True)
            df = df.reset_index(drop=True)
            df.to_csv(exportPath + f'/{f}.csv')
            
            qq = qq + 1
            # if qq == 2:
            #     pdb.set_trace()
            
        

# path = r'F:\Ahmed/'
# action_path = 'D:\PhD Edinburgh\RealSenseTests\EatSense/actionlabels'
# exportPath = 'D:\PhD Edinburgh\Codes\converters\csvTOcoco/seqsExports'
path = '/home/tkfpsk/minf2/dataset'
# getActions(action_path, '20210523_202300')
getPoses(path,'20210923_161107')
# getEverything(path, action_path, exportPath)