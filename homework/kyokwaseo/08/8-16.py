#요소추가
#☆☆☆☆append 사용 ☆☆☆☆☆


import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)

#인덱스가 "pineapple"이고, 데이터가 12인 요소를 series에 추가해라.

x = pd.Series([12], index = ["pineapple"])
series = series.append(x)

print(series)

'''
apple        10
orange        5
banana        8
strberry     12
kiwi          3
pineapple    12
dtype: int64
'''

#추가완료

