import pandas as pd
import re
import numpy as np
import cv2
import os
import subprocess
from os import listdir
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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

def draw_lines(frame, coordinates, img_name):
    '''
    draw a skeleton using keypoint pairs
    '''
    for i in keypoint_pairs:
        point_j1 = np.asarray(coordinates[i[0]].astype(int))
        point_j2 = np.asarray(coordinates[i[1]].astype(int))
        cv2.circle(frame, point_j1, 10, (0,255,0), -1)
        cv2.circle(frame, point_j2, 10, (0,255,0), -1)
        cv2.line(frame, point_j1, point_j2, (0,255,0), 3)
        
    cv2.imwrite(img_name, frame)

def euclidean_distance(before_dataframe, after_dataframe, type='2d'):
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
            ab = np.vstack(a-b)
            abn = np.linalg.norm(ab, axis=1)
            # max frame
            # m = before_dataframe['id'][np.argmax(abn)]
            m = np.argmax(abn)
            print(a)
            print(b)
            
            
            # gather x, y coordinates of the frame
            raw_rows.append({'Keypoint': i, 'Frame': m, 'Distance': np.around(np.max(abn), 3),'x before': np.around(b.iloc[m][0], 3), 'x after': np.around(a.iloc[m][0], 3), 'y before': np.around(b.iloc[m][1], 3), 'y after': np.around(a.iloc[m][1], 3)})

            # calculate mean, max, min, STD and add to the dataframe
            rows.append({'Keypoint': i, 'Median': np.around(np.median(abn), 3),'Mean': np.around(np.mean(abn), decimals=3), 'Max': np.around(np.max(abn), decimals=3),'Min': np.around(np.min(abn), decimals=3), 'STD': np.around(np.std(abn), decimals=3)})
            # print(i, 'max frame:', np.around((np.argmax(abn)/900)), np.argmax(abn)%900/15)
        df = pd.DataFrame(data=rows ,columns=idx)
        raw_df = pd.DataFrame(data=raw_rows ,columns=raw_idx)
        return df, raw_df

    elif (type == '3d'):
        
        return
    else:
        print('wrong type')
            

# raw_df.to_csv('raw_positions.csv')
# euclidean_distance().to_csv('euclidean_distance.csv')

if __name__ == "__main__":
    # before = './output/pose/20220812_124345/20220812_124345_preds_HigherHRNet.csv'
    # after = './output/fake_pose/20220812_124345/20220812_124345_preds_HigherHRNet.csv' 
    # df_b = csv_to_df(before)
    # df_a = csv_to_df(after)
    # ed_df, raw_df = euclidean_distance(df_b, df_a)

    # print(ed_df)
    # print()
    # print(raw_df)
   
    # for idx in raw_df.index:
    #     frame = raw_df['Frame'][idx]
    #     image = get_frame(a_path, raw_df['Frame'][idx])
    #     name = 'output/'+raw_df['Keypoint'][idx] + '.png'
    #     draw_lines(image, df_a.iloc[frame], name)
    #     bname = 'output/'+raw_df['Keypoint'][idx] + '_b.png'
    #     bimage = get_frame(a_path, raw_df['Frame'][idx])
    #     draw_lines(bimage, df_b.iloc[frame], bname)

    b = '/mnt/c/Users/tkfps/Downloads/b/'
    a = '/mnt/c/Users/tkfps/Downloads/a/'
    check_integrity(b, a)
    
    
    # img_to_vid('/mnt/c/Users/tkfpsk/Downloads/20220812_124345/%010d.png', '/mnt/c/Users/tkfpsk/Downloads/output/20220812_124345.mp4')

    # os.system("ffmpeg -r 30 -i /mnt/c/Users/tkfpsk/Downloads/20220812_124345/%010d.png -vcodec mpeg4 -y /mnt/c/Users/tkfpsk/Downloads/output/20220812_124345.mp4")