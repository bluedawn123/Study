#5, 6 

import pandas as pdf

indexs = ["apple", "orange", "banana", "strberry", "kiwi"]
data = [10, 5, 8, 12, 3]

series = pdf.Series(data, index = indexs)


print(series)

#맘대로 해줘도 맞는다. 중요한 것은 Series()의 의미