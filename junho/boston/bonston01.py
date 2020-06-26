import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_boston

boston = load_boston()

print(boston)
print("데이터의 키 값 : ", boston.keys())  #'data', 'target', 'feature_names', 'DESCR', 'filename'

x = boston.data          #x,y를 어떻게 설정하고 data와 target은 어떻게 설정하는가 ?!?!? 위 참조
y = boston.target 
z = boston.feature_names
print(x.shape)           # (506, 13)
print(y.shape)           # (506,)
print("보스톤의 feature_names : ", z)  #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM'
                                       #'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']


