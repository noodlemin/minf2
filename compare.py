import pandas as pd
import re
import numpy as np
import cv2

# 8 column names for all csv files
col_names = ['head', 'Mid-Shoulder', 'Right-Shoulder', 'Right-Elbow', 'Right-Wrist', 'Left-shoulder', 'Left-Elbow', 'Left-Wrist']

# preprocessing
def csv_to_df(csv_file):
    '''
    Convert a csv file to a dataframe.

    Returns: a dataframe
    '''
    df  = pd.read_csv(csv_file)
    # regular expression to remove brackets and comma
    lat_lon_re = re.compile(r'\((.*)?\,(.*)\)')
    # convert strings to tuple
    for i in col_names:
        df[i] = [np.array(list(map(float,lat_lon_re.findall(p)[0]))) for p in df[i]]

# print(after_df['imgName'][0])
# check types
# for i in col_names:
#     print(type(before_df[i][0]))
#     print(type(after_df[i][0]))
# print(type(before_df['image_id'][0]))

# check whether two dataframes are equal
# print(before_df['image_id'].equals(after_df['image_id']))

# extract the frame and save
a_path = 'vis_20210923_161107_a.mp4'
b_path = 'vis_20210923_161107_b.mp4'
def get_frame(path, name, frame):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
    res, frame = cap.read()
    cv2.imwrite(name+'_a.jpg', frame)

'''
Euclidean Distance
'''
def euclidean_distance(before_dataframe, after_dataframe):
    idx = ['Keypoint','Median','Mean', 'Max', 'Min', 'STD']
    raw_idx = ['x before', 'x after', 'y before', 'y after']
    # for each key point
    rows = []
    raw_rows = []
    for i in col_names:
        # get norm value of a-b
        a = after_dataframe[i]
        b = before_dataframe[i]
        ab = np.vstack(a-b)
        abn = np.linalg.norm(ab, axis=1)
        # max frame
        m = np.argmax(abn)
        # get_frame(a_path, i, m)
        
        # gather x, y coordinates of the frame
        raw_rows.append({'x before': np.around(b[m][0], 2), 'x after':np.around(a[m][0], 2), 'y before':np.around(b[m][1], 2), 'y after':np.around(a[m][1], 2)})

        # calculate mean, max, min, STD and add to the dataframe
        rows.append({'Keypoint': i, 'Median':np.around(np.median(abn), 3),'Mean':np.around(np.mean(abn), decimals=3), 'Max': np.around(np.max(abn), decimals=3),'Min': np.around(np.min(abn), decimals=3), 'STD': np.around(np.std(abn), decimals=3)})
        print(i, 'max frame:', np.around((np.argmax(abn)/900)), np.argmax(abn)%900/15)
    df = pd.DataFrame(data=rows ,columns=idx)
    raw_df = pd.DataFrame(data=raw_rows ,columns=raw_idx)
    return df, raw_df


# raw_df.to_csv('raw_positions.csv')
# euclidean_distance().to_csv('euclidean_distance.csv')

if __name__ == "__main__":
    before = 'before.csv'
    after = 'after.csv' 
    df  = pd.read_csv(before)
    # save 8 column names
    col_names = list(df.columns.values)[2:10]
    print(col_names)
    # df, raw_df = euclidean_distance(before_df, after_df)