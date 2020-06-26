import pandas as pd

index = ["apple", "orange", "banana", "strberry", "kiwi"]

data = [10, 5, 8, 12, 3]

series = pd.Series(data, index=index)

data = {"fruits" : ["apple", "orange", "banana", "strberry", "kiwi"],
        "year" : [2001, 2002, 2001, 2008, 2006],
        "time" : [1, 4, 5, 6, 3]
}

df = pd.DataFrame(data)

print("series 데이터 : ", series)
print("  ")
print("dataframe 데이터")
print(df)

'''
#Series는 1차원 배열을 다룰 수 있다.
pandas.Series(딕셔너리형태의 리스트)   예를들어 (data, index=index)형태면, 리스트를 생성해서 Series에 전달가능.







'''