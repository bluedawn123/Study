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


wine = pd.read_csv('./data/csv/winequality-white.csv', index_col = None, header=0, encoding='cp949', sep=';') 

print("wine data : ", wine)
print("wine데이터의 형태 : ", wine.shape)         #(4898, 11)

print("wine.keys() : ", wine.keys())   #12개의 키값. 

x = wine
y = wine.quality


