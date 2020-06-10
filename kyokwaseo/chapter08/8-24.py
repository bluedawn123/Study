#정렬
import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)

conditions = [True, True, False, False, False]
#print(series[conditions])

#문제. series의 요소에를 알파벳순 정렬해서 a에 대입
a = series.sort_index()   #(sort_index)함수

#문제. series의 데이터를 오름차순으로 정렬해서 b에 대입

b = series.sort_values()

print(a)
print(b)

'''
apple       10
banana       8
kiwi         3
orange       5
strberry    12
dtype: int64
kiwi         3
orange       5
banana       8
apple       10
strberry    12
dtype: int64
'''












