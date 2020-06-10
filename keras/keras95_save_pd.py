#아이리스

import numpy as np
import pandas as pd


datasets = pd.read_csv('./data/csv/iris.csv',
                        index_col=None, 
                        header = 0, sep = ',')  #header = 0으로 하는 이유는 데이터셋의 데이터 부분을 맞추기 위해
'''
0    5.1  3.5     1.4         0.2          0
1    4.9  3.0     1.4         0.2          0
..   ...  ...     ...         ...        ...
148  6.2  3.4     5.4         2.3          2
149  5.9  3.0     5.1         1.8          2  이 부분이 실질적인 데이터 이다. 
'''

print(datasets)
print("datasets의 형태 : ", datasets.shape)

print(datasets.head())  #위에서 5열 정도
print(datasets.tail())  #아래서 5열


print(datasets.values)  # 값들만 출력 ☆☆☆판다스를 numpy로 바꾼다 ☆☆☆

aaa = datasets.values
print(type(aaa))     # 값들만 출력 ☆☆☆판다스를 numpy로 바꾼다 ☆☆☆   


#ㅡㅡㅡㅡㅡㅡㅡㅡ헤더와 인덱스를 제거하고 실질적인 데이터를 numpy로 바꾸는 과정ㅡㅡㅡㅡㅡㅡㅡㅡㅡ


#넘파이로 저장하시오 

np.save('./data/iris_save.npy', arr=aaa)



