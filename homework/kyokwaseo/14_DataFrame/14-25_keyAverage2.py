#24번의 magnesium 평균값 출력

import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", 
                    header = None)
df.columns=[" ", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
            "Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

print(df)
print("-----------------")
print("마그네슘의 평균 : ", df["Magnesium"].mean())
print("데이터의 열 출력 : ", df.columns)

