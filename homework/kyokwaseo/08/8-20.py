
import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)

conditions = [True, True, False, False, False]
#print(series[conditions])

print(series[series >= 5])

print("dddddddddddddddddddddddddd")

print(series[series >= 10])

'''
apple       10
orange       5
banana       8
strberry    12
dtype: int64
dddddddddddddddddddddddddd
apple       10
strberry    12
dtype: int64
PS D:\Study>
'''