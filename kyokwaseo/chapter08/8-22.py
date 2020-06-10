import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)

conditions = [True, True, False, False, False]
#print(series[conditions])

#문제. series의 요소에서 5이상 10미만을 포함하는 series를 새로 만들어 series에 다시 대입하라


series = series[series>=5][series < 10]

print(series)

'''
orange    5
banana    8
dtype: int64
'''