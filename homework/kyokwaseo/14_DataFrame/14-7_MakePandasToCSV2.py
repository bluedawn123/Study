import pandas as pd

data = {"OS": ["Machintosh", "Windows", "Linux"],
        "release": [1984, 1985, 1991],
        "country": ["US", "US", ""]}


df = pd.DataFrame(data)
print(df)


df.to_csv("OSlist.csv")