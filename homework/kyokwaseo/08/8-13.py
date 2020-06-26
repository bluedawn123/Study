import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)


#1. series_values에 series 값을 대입하라
series_values = series.values          #규칙임

#2. series_indexes의 인덱스를 대입하라
series_index = series.index            #규칙임

print(series_values)
print(series_index)


'''
[10  5  8 12  3]
Index(['apple', 'orange', 'banana', 'strberry', 'kiwi'], dtype='object')
'''





