import pandas as pd

fruits = {"banana" : 3, "orange": 4, "grape" : 1, "peach" : 5}
series = pd.Series(fruits)

#☆☆☆Series는 리스트 형처럼 자를 수도 있다.

print(series[0:2])

'''
banana     3
orange     4
dtype: int64
'''


print(series[["orange", "peach"]])

'''
orange    4
peach     5
dtype: int64
PS D:\Study>
'''

