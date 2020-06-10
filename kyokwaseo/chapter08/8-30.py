#행추가datetime A combination of a date and a time. Attributes: ()


import pandas as pd

data = {"fruits" : ["apple", "orange", "banana", "strberry", "kiwi"],
        "time " : [1, 4, 5, 6, 3]}


df = pd.DataFrame(data)
print(df)

series = pd.Series(["mango", 2008, 7], index = ["fruits", "year", "time"])

df2 = df.append(series, ignore_index = True)

print(df2)