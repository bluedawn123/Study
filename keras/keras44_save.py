import numpy as np
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

#2. 모델구성
model = Sequential()
model.add(LSTM(10, input_shape = (4,1)))
model.add(Dense(11))
model.add(Dense(1))
model.summary()

#model.save(".//model//save_44.h5")  모델만 저장!!
model.save(".//model//save_45.h5")
print("저장 잘됐다.")