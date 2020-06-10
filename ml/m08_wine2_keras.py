import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler      # (x - 최소) / (최대 - 최소)
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'

cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto', save_weights_only = False, verbose = 1)
'''


wine = pd.read_csv('./data/csv/winequality-white.csv', index_col = 0, header=0, encoding='cp949', sep=';') 

  
#samsung = samsung.dropna()#삼성 nan 값 제거
#hite = hite.dropna() #하이트 nan 제거

print("wine data : ", wine)
print("wine데이터의 형태 : ", wine.shape)         #(4898, 11)

print("wine.keys() : ", wine.keys())   #11개의 키값. 


print("  ")
print("------------------이하 넘파이 변경 후-----------------------")
print("중요한 것은 넘파이 변경 후 우리가 수업대로 쓸 수 있도록 변경 된다는 것이다.")

wine = wine.values
print("wine의 타입 : ", type(wine))  #넘파이로 저장 완료된 걸 확인할 수 있다. 즉 넘파이형 저장완료
np.save('./data/wine.npy', arr=wine)
wine = np.load('./data/wine.npy', allow_pickle=True)
print("wine의 형태 : ", wine.shape)  #(4898, 11)





