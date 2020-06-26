#딸기 요소삭제
import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
datas = [10, 5, 8, 12, 3]
series = pd.Series(datas, index = indexes)

#index와 data를 토함한 series를 작성하여 series에 대입한다

print(series)
'''
apple       10
orange       5
banana       8
strberry    12
kiwi         3
dtype: int64
'''

series = series.drop("strberry")
print(series)
'''
apple     10
orange     5
banana     8
kiwi       3
dtype: int64
'''

