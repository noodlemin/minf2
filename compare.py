import pandas as pd

before  = pd.read_csv('before.csv')
after  = pd.read_csv('after.csv')

col_names = []
for i in before.columns:
    col_names.append(i)



cols = col_names[2:10]
before[cols] = before[cols].astype(float)

print(type(before['head'][0]))
# print(after['head'].head())


# for i in 