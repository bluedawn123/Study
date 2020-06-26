import pandas as pd

# 불러오기
df = pd.read_csv(
"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None)

#형태 한번 확인

print(df)   #(150, 5) 아직 데이터만 있고 컬럼명은 없다

#컬럼명을 아래와 같이 지정해준다. 
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

df