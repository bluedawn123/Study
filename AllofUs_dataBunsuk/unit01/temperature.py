import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import missingno as msno

seoulweather = pd.read_csv('./data/bunsukcsv/seoul.csv', header = 0, 
                            index_col=0, encoding='CP949')

print("아래는 seoulweather에 대한 정보들")
print("  ")
print("seoulweather.shape : ", seoulweather.shape)
z = seoulweather.columns
print("컬럼들 : ", z)   # Index(['지점', '평균기온(℃)', '최저기온(℃)', 
                       #'최고기온(℃)'], dtype='object')

print(seoulweather)

print("----------------------------------")

