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
    idx = ['Keypoint','Mean', 'Max', 'Min', 'STD']
    # for each key point
    rows = []
    for i in col_names:
        # get norm value of a-b
        a = after_df[i]
        b = before_df[i]
        ab = np.vstack(a-b)
        abn = np.linalg.norm(ab, axis=1)
         
        # calculate mean, max, min, STD and add to the dataframe
        rows.append({'Keypoint': i,'Mean':np.mean(abn), 'Max': np.max(abn),'Min': np.min(abn), 'STD': np.std(abn)})
    df = pd.DataFrame(data=rows ,columns=idx)
    return df

print(euclidean_distance())

