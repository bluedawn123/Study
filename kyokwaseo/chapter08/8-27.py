import pandas as pd

indexes = ["apple", "orange", "banana", "strberry", "kiwi"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]

series1 = pd.Series(data1, index = indexes)
series2 = pd.Series(data2, index = indexes)

#문제. series1, series2로 dataframe을 생성하여 df에 대입하고 출력하라.

df = pd.DataFrame([series1, series2])
df2 = pd.DataFrame([series1], series2)
df3 = pd.DataFrame([series1])
print("df : ",df)
print(" ")
print("df2 : ",df2)
print(" ")
print("df3 : ", df3)















