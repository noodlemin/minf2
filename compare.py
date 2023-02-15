import pandas as pd
import re
import numpy as np
import cv2
import os
import subprocess
from os import listdir
# from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import functools
import operator
import collections
import tqdm
import copy

# 8 column names for all csv files
col_names = ['head', 'Mid-Shoulder', 'Right-Shoulder', 'Right-Elbow', 'Right-Wrist', 'Left-shoulder', 'Left-Elbow', 'Left-Wrist']
# joint pairs to draw pose estimation skeletons
keypoint_pairs = [['head', 'Mid-Shoulder'], ['Mid-Shoulder', 'Right-Shoulder'], ['Right-Shoulder', 'Right-Elbow'], ['Right-Elbow', 'Right-Wrist'], ['Mid-Shoulder', 'Left-shoulder'], ['Left-shoulder', 'Left-Elbow'], ['Left-Elbow', 'Left-Wrist']]

# test video path
a_path = 'deepfaked/20210923_161107.mp4'

# pick one video from each person
videos = []

def img_to_vid(img_folder, vid_path):
    os.system('ffmpeg -f image2 -i ' + img_folder + ' -vcodec mpeg4 -y ' + vid_path)


def Diff_img(img0, img):
  '''
  This function is designed for calculating the difference between two
  images. The images are convert it to an grey image and be resized to reduce the unnecessary calculating.
  '''
  try:
    # Grey and resize
    img0 =  cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    img =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img0 = cv2.resize(img0, (320,200), interpolation = cv2.INTER_AREA)
    img = cv2.resize(img, (320,200), interpolation = cv2.INTER_AREA)
    # Calculate
    Result = (abs(img - img0)).sum()
    return Result
  except:
    print('the heck?')

# checking integrity
dpath = '/mnt/c/Users/tkfps/Downloads/why/'
def check_integrity(before_path, after_path):
    before_list = listdir(before_path)
    after_list = listdir(after_path)
    count = 0
    for i in range(len(before_list)):
        b = cv2.VideoCapture(before_path+before_list[i])
        a = cv2.VideoCapture(after_path+after_list[i])
        b_length = int(b.get(cv2.CAP_PROP_FRAME_COUNT))
        a_length = int(a.get(cv2.CAP_PROP_FRAME_COUNT))

        if b_length != a_length:
            count +=1
            print(b_length)
            print(a_length)
            inputfilename = after_path+after_list[i]
            print(inputfilename)
            outputfilename = '/mnt/c/Users/tkfps/Downloads/output/' + after_list[i]
            frame = str(a_length-1) # has to be a string
            subprocess.call(['ffmpeg', '-i', inputfilename, '-frames:v', frame, outputfilename])
            # break
            # print('diff:', a_length - b_length)
            # cv2.imwrite(dpath+after_list[i]+'_0.png', get_frame(after_path+after_list[i], a_length))
            # cv2.imwrite(dpath+after_list[i]+'_1.png', get_frame(after_path+after_list[i], a_length-1))
            # cv2.imwrite(dpath+after_list[i]+'_2.png', get_frame(after_path+after_list[i], a_length-2))
            # cv2.imwrite(dpath+after_list[i]+'_3.png', get_frame(after_path+after_list[i], a_length-3))
            # b_i = np.asarray(b_image)
            # a_i = np.asarray(a_image)
            # print(b_image.shape)
            # print(a_image.shape)
            # simlarityIndex = ssim(b_i, a_i)
            #  data_range=a_image.max() - a_image.min())
            # print(simlarityIndex)
            # if b_length != a_length:
            #     print(after_path+after_list[i])
            #     print('diff:', a_length - b_length)
            # print('after:', )

            # ret,frame0 = a.read()
            # # Result = []
            # Num = 0
            # for i in range(a_length-1):
            #     ret,frame=a.read()
            #     # print(frame)
            #     #cv2.imshow("video",frame)
            #     if Num > 0:
            #         diff = Diff_img(frame0, frame)
            #         frame0 = frame
            #         if diff < 10000:
            #             print(Num)
            #     Num += 1
            #     if cv2.waitKey(25)&0xFF==ord('q'):
            #         cv2.destroyAllWindows()
            #         break
            # a_length-1
            # a.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # _, a_image = a.read()
            # aa = cv2.VideoCapture(after_path+after_list[i])
            # aa.set(cv2.CAP_PROP_POS_FRAMES, 1)            
            # _, aa_image = aa.read()
            # print(after_path+after_list[i])
            # print(Diff_img(a_image, aa_image))
            # cv2.imshow('image b', aa_image)
            # cv2.imshow('image a', aaa_image)
            # cv2.waitKey(0)
            # plt.show()
        #     for i in range(b_length):
                # _, b_image = b.read()
                # _, a_image = a.read()
                # b_i = np.squeeze(b_image)
                # a_i = np.squeeze(a_image)
                
                # print(simlarityIndex)
            # print(before_list[i])
            # print(b_length)
            # print(after_list[i])
            # print(a_length)
    print(count)
# def sim_check(b, a):
#     simlarityIndex = ssim(b, a)


# preprocessing
def csv_to_df(csv_file):
    '''
    Convert a csv file to a dataframe.

    Returns: a dataframe
    '''
    df = pd.read_csv(csv_file)
    # to store the coodinates
    df_new = pd.DataFrame(columns=col_names)
    # regular expression to remove brackets and comma
    lat_lon_re = re.compile(r'\((.*)?\,(.*)\)')
    # convert strings to tuple
    for i in col_names:
        df_new[i] = [np.array(list(map(float,lat_lon_re.findall(p)[0]))) for p in df[i]]
    # add 'image_id' column
    df_new.insert(0, "id", df['image_id'])
    # sort the dataframe by id
    df_new = df_new.sort_values(by=['id'])
    return df_new

def csv3d_to_df(csv_file):
    return pd.read_csv(csv_file)


# print(after_df['imgName'][0])
# check types
# for i in col_names:
#     print(type(before_df[i][0]))
#     print(type(after_df[i][0]))
# print(type(before_df['image_id'][0]))

# check whether two dataframes are equal
# print(before_df['image_id'].equals(after_df['image_id']))


def get_frame(path, frame):
    '''
    extract the frame from the video and return it

    Returns: a frame from the video
    '''
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
    res, frame = cap.read()
    return frame

def draw_lines(frame, coordinates, img_name, color):
    '''
    draw a skeleton using keypoint pairs
    '''
    if color == 'red':
        rgb = (0, 0, 255)
    elif color == 'green':
        rgb = (0, 255, 0)
    for i in keypoint_pairs:
        point_j1 = np.asarray(coordinates[i[0]].astype(int))
        point_j2 = np.asarray(coordinates[i[1]].astype(int))
        cv2.circle(frame, point_j1, 10, rgb, -1)
        cv2.circle(frame, point_j2, 10, rgb, -1)
        cv2.line(frame, point_j1, point_j2, rgb, 3)
        
    cv2.imwrite(img_name, frame)

exception_dataframe = pd.read_csv('/Users/min/minf2/data_15classNother_v5/df_all.csv')
def euclidean_distance(data, before_dataframe, after_dataframe, project_name, type='2d'):
    '''
    Returns: 
        df: Euclidean distance dataframe 
        raw_df: Frames and coordinates of keypoints where the Euclidean distance is maximum 
    '''
    idx = ['Keypoint','Median','Mean', 'Max', 'Min', 'STD']
    raw_idx = ['Keypoint', 'Frame', 'Distance', 'x before', 'x after', 'y before', 'y after']
    # for each key point
    rows = []
    raw_rows = []
    
    if type == '2d':
        for i in col_names:
            # get norm value of a-b
            a = after_dataframe[i]
            b = before_dataframe[i]
            # ab = np.vstack(a-b)
            # abn = np.linalg.norm(ab, axis=1)
            # max frame
            # m = before_dataframe['id'][np.argmax(abn)]
            abn = data[i]
            # want to maintain original indices, so store originl in the dictionary
            temp_dict = {}
            for j, dist in enumerate(abn):
                temp_dict[dist] = j
            # m = np.argmax(abn)

            # remove 'other' action frames
            
            # try:
            #     abn = np.delete(abn, ex_list)
            # except:
                
            m = temp_dict.get(np.max(abn))

            

            

            # gather x, y coordinates of the maximum distance frame
            raw_rows.append({'Keypoint': i, 'Frame': m, 'Distance': np.around(np.max(abn), 3),'x before': np.around(b.iloc[m][0], 3), 'x after': np.around(a.iloc[m][0], 3), 'y before': np.around(b.iloc[m][1], 3), 'y after': np.around(a.iloc[m][1], 3)})

            # calculate mean, max, min, STD and add to the dataframe
            rows.append({'Keypoint': i, 'Median': np.around(np.median(abn), 3),'Mean': np.around(np.mean(abn), decimals=3), 'Max': np.around(np.max(abn), decimals=3),'Min': np.around(np.min(abn), decimals=3), 'STD': np.around(np.std(abn), decimals=3)})
            # print(i, 'max frame:', np.around((np.argmax(abn)/900)), np.argmax(abn)%900/15)
        df = pd.DataFrame(data=rows ,columns=idx)
        raw_df = pd.DataFrame(data=raw_rows ,columns=raw_idx)
        print(raw_df)
        return df, raw_df

    elif (type == '3d'):
        return
    else:
        print('wrong type')

def prepro_data(before_path, after_path, fps, project_name):
    # keypoints dictionaries
    freq_dict = []
    # Euclidean distance dictionary
    ed_dict = dict.fromkeys(col_names, [])
    
    before_dataframe = csv_to_df(before_path)
    after_dataframe = csv_to_df(after_path)
    if before_dataframe.shape != after_dataframe.shape:
        print('the heck?')
    # pd.set_option('display.max_columns', None)
    # print(before_dataframe.head())
    # print(after_dataframe.head())
    # for each key point

    # project_name = 
    for i in col_names:
        # get norm value of a-b
        # b = before_dataframe[[i, 'id']]
        # a = after_dataframe[[i, 'id']]
        a = after_dataframe[i]
        b = before_dataframe[i]
        ab = np.vstack(a-b)
        temp_eds = np.linalg.norm(ab, axis=1)
        ex_list = exception_list(project_name)
        eds = []
        if fps == 30:
            for j, val in enumerate(temp_eds):
                if j%2 == 0:
                    eds.append(val)
            new_ex_list = np.divide(ex_list, 2).astype(int)
            try:
                eds = np.delete(eds, new_ex_list)
            except:
                print(new_ex_list[0])
                # print(ex_list[0])
                exit()
            ed_dict[i] = eds
        else:
            eds = np.delete(temp_eds, ex_list)
            ed_dict[i] = eds
        # print(i, len(ed_dict[i]))

        # sort by image_id
        # b.sort_values(by=['image_id'])
        # a.sort_values(by=['image_id'])
        # merged_df = pd.merge(b, a, on =['id'])
        # dist = (merged_df[i+'_x'] - merged_df[i+'_y']).tolist()
        # print(dist)
        # dist = np.round(np.linalg.norm(dist, axis=1), 0).astype(int)
        dist = np.round(eds, 0).astype(int)
        temp = {}
        for j in dist:
            if j in temp:
                temp[j] += 1
            else:
                temp[j] = 1
        freq_dict.append(temp)
        # ed_dict.append(ed_temp)

        # key_dict[i].append(dist)
    return freq_dict, ed_dict

def max_ed_images(after_dataframe, after_video_path, before_dataframe, raw_dataframe, outpath):
    for idx in raw_dataframe.index:
        frame = raw_dataframe['Frame'][idx]
        image = get_frame(after_video_path, raw_dataframe['Frame'][idx])
        name = 'output/'+ outpath + raw_dataframe['Keypoint'][idx] + '.png'
        draw_lines(image, before_dataframe.iloc[frame], name, 'green')
        draw_lines(image, after_dataframe.iloc[frame], name, 'red')
        # bname = 'output/'+raw_dataframe['Keypoint'][idx] + '_b.png'
        # bimage = get_frame(after_video_path, raw_dataframe['Frame'][idx])


# remove frames at the beginning or the end of the video
label_path = '/Users/min/minf2/data_15classNother_v5/'
split_list = ['sp1.csv', 'sp2.csv', 'sp3.csv', 'sp4.csv', 'sp5.csv']
# action 0: other
ignore_action_list = [0]
required_columns = ['Project', 'Imgs', 'Action', 'fps']
after_dir = listdir('/Users/min/minf2/2dout/')
after_dir.remove('.DS_Store')

def exceptions():
    dfs = []
    print('processing CSVs')
    for i in tqdm.tqdm(split_list):
        temp_df = pd.read_csv(label_path+i)
        # print(temp_df['Action'].dtypes)
        # break
        df = temp_df[required_columns]
        df = df[df['Project'].isin(after_dir)]
        df = df[df['Action'].isin(ignore_action_list)]
        dfs.append(df)
    concated_df = pd.concat(dfs, axis=0)
    concated_df.to_csv(label_path+'df_all.csv')

# 
def exception_list(project_name):
    # print(exception_dataframe.shape)
    df = exception_dataframe[exception_dataframe['Project']==project_name]
    # print(df.shape)
    exception_list = df['Imgs'].tolist()
    return exception_list






    



# raw_df.to_csv('raw_positions.csv')
# euclidean_distance().to_csv('euclidean_distance.csv')

if __name__ == "__main__": 

    before_path = '/Users/min/minf2/2dori/'
    after_path = '/Users/min/minf2/2dout/'
    before_dir = listdir('/Users/min/minf2/2dori/')
    after_dir = listdir('/Users/min/minf2/2dout/')
    after_dir.remove('.DS_Store')
    image_path = '/Users/min/minf2/output/images/'
    csv_path = '/Users/min/minf2/output/csvs/'
    freq_dicts = []
    ed_dicts = []
    
    # exceptions()

    print('Processing')
    for i in tqdm.tqdm(after_dir):
        if i == '.DS_Store':
            continue
        temp_fps = exception_dataframe[exception_dataframe['Project']==i].iloc[0]
        fps = temp_fps['fps']
        print(i, fps)
        file_name = i + '_preds_HigherHRNet.csv'
        freq_dict, ed_dict= prepro_data(before_path+file_name, after_path+i+'/'+file_name, fps, i)
        freq_dicts.append(freq_dict)
        ed_dicts.append(ed_dict)

    # sum frequencies and ED
    freq_sum = []
    ed_sum = []

    for i in range(len(freq_dicts)):
        freq_project = freq_dicts[i]
        ed_project = ed_dicts[i]
        if i == 0:
            freq_sum = freq_project
            ed_sum = ed_project
        else:
            for j in range(len(freq_sum)):
                freq_sum[j] = {k: freq_project[j].get(k, 0) + freq_sum[j].get(k, 0) for k in set(freq_project[j]) | set(freq_sum[j])}
            for j, ed in enumerate(ed_dicts):
                if j == 0:
                    ed_sum = ed.copy()
                else:
                    temp_list = []
                    for key in ed:
                        temp_list = np.concatenate((ed_sum[key], ed[key]), axis=None)
                        ed_sum[key] = temp_list


                
                        
            # for j, key in enumerate(ed_sum):
                # ed_sum[j] = {k: ed_project[j].get(k, 0) + ed_sum[j].get(k, 0) for k in set(ed_project[j]) | set(ed_sum[j])}
    
    # get images and statistics
    for i, dict in enumerate(ed_dicts):
        file_name = after_dir[i] + '_preds_HigherHRNet.csv'
        before_dataframe = csv_to_df(before_path+file_name)
        after_dataframe = csv_to_df(after_path+after_dir[i]+'/'+file_name)
        df, raw_df = euclidean_distance(dict, before_dataframe, after_dataframe, after_dir[i])
        df.to_csv(csv_path+after_dir[i]+'.csv')
        os.makedirs('/Users/min/minf2/output/' + after_dir[i])
        for idx in raw_df.index:
            vid_path = '/Users/min/Desktop/deepfaked_videos/' + after_dir[i] + '.mp4'
            outpath = after_dir[i] + '/'
            max_ed_images(after_dataframe, vid_path, before_dataframe, raw_df, outpath)


            # frame = raw_df['Frame'][idx]
            # image = get_frame(vid_path, raw_df['Frame'][idx])
            # name = 'output/'+ after_dir[i] + '/'+ raw_df['Keypoint'][idx] + '.png'
            # draw_lines(image, after_dataframe.iloc[frame], name)
            # bname = 'output/'+ after_dir[i] + '/' + raw_df['Keypoint'][idx] + '_b.png'
            # bimage = get_frame(vid_path, raw_df['Frame'][idx])
            # draw_lines(bimage, before_dataframe.iloc[frame], bname)
        
    

    # draw histogram
    for n, i in enumerate(freq_sum):
        plt.clf()
        plt.bar(range(len(i)), list(i.values()), tick_label=list(i.keys()))
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks()[::10])
        plt.xlabel('Euclidean Distance (pixel)')
        plt.ylabel('Frequency')
        plt.title(col_names[n])
        plt.savefig(image_path+col_names[n]+'.png')
        
    



    # check_integrity(b, a)
    # get_frame(path, frame)