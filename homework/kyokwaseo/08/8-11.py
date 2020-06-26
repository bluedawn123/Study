import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)  #여기서 입력을 받는다는 게 중요. 전환

items1 = series[1:4]
items2 = series[["apple", "banana", "kiwi"]]

print(items1)
print(items2)

'''
orange       5
banana       8
strberry    12
dtype: int64
apple     10
banana     8
kiwi       3
dtype: int64
'''