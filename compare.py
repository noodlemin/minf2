import pandas as pd
import re
import numpy as np

'''
preprocessing
'''
# csv to dataframe
before = 'before.csv'
after = 'after.csv'
before_df  = pd.read_csv(before)
after_df  = pd.read_csv(after)
# save 8 column names
col_names = list(before_df.columns.values)[2:10]
# regular expression to remove brackets and comma
lat_lon_re = re.compile(r'\((.*)?\,(.*)\)')
# convert string to tuple
for i in col_names:
    before_df[i] = [np.array(list(map(float,lat_lon_re.findall(p)[0]))) for p in before_df[i]]
    after_df[i] = [np.array(list(map(float,lat_lon_re.findall(p)[0]))) for p in after_df[i]]
    # before_df[i] = [tuple(reversed(list(map(float,lat_lon_re.findall(p)[0])))) for p in before_df[i]]
    # after_df[i] = [tuple(reversed(list(map(float,lat_lon_re.findall(p)[0])))) for p in after_df[i]]

# print(after_df['imgName'][0])
# check types
# for i in col_names:
#     print(type(before_df[i][0]))
#     print(type(after_df[i][0]))
# print(type(before_df['image_id'][0]))

# check whether two dataframes are equal
# print(before_df['image_id'].equals(after_df['image_id']))

# create the dataframe

'''
Euclidean Distance
'''
def euclidean_distance():
    idx = ['Keypoint','Median','Mean', 'Max', 'Min', 'STD']
    # for each key point
    rows = []
    for i in col_names:
        # get norm value of a-b
        a = after_df[i]
        b = before_df[i]
        ab = np.vstack(a-b)
        abn = np.linalg.norm(ab, axis=1)
         
        # calculate mean, max, min, STD and add to the dataframe
        rows.append({'Keypoint': i, 'Median':np.around(np.median(abn), 3),'Mean':np.around(np.mean(abn), decimals=3), 'Max': np.around(np.max(abn), decimals=3),'Min': np.around(np.min(abn), decimals=3), 'STD': np.around(np.std(abn), decimals=3)})
        print(i, 'max frame:', np.around((np.argmax(abn)/900)), np.argmax(abn)%900/15)
    df = pd.DataFrame(data=rows ,columns=idx)
    return df

print(euclidean_distance())
# euclidean_distance().to_csv('euclidean_distance.csv')

