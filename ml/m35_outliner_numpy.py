# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구하시오.
import numpy as np
import pandas as pd

def outliers(a1):
    
    outliers = []
    for i in range(a1.shape[1]):
        print("a1.shape[1] : ", a1.shape[1])  #2
        data = a1[:, i]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        
        outliers.append(out)
    return outliers


a2 = np.array([[1, 5000],[200, 8],[2, 4],[3, 7],[8, 2]])
print("a2의 shape : ", a2.shape)  #(5, 2)
print(a2)

# [[   1 5000]
#  [ 200    8]
#  [   2    4]
#  [   3    7]
#  [   8    2]]

b2 = outliers(a2)
print(b2)
# [(array([1], dtype=int64),), (array([0], dtype=int64),)]   
#    200 이 이상치,                5000이 이상치

