import pandas as pd
import re
import numpy as np

'''
preprocessing
'''
# csv to dataframe
before_df  = pd.read_csv('before.csv')
after_df  = pd.read_csv('after.csv')
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

'''
Euclidean Distance
'''
dist = []
a = after_df['head']
b = before_df['head']
# point1 = np.array((1, 2, 3))
# print(a[0]-a[1])
# print(type(before_df['head'][0]))
# dist = np.linalg.norm(a - b)
# print(dist)
for i in range(len(a)):
    dist.append(np.linalg.norm(a[i] - b[i]))
# print(dist)
print(np.max(dist))