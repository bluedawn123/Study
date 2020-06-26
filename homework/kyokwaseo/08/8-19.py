#필터링

'''
씨리즈형 데이터에서 조건과일치하는 요소를 꺼내고 싶을때 사용.
pandas에서는 bool형의 시퀀스를 지정해서 true인 것만 추출가능.
'''



import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)

conditions = [True, True, False, False, False]
print(series[conditions])

