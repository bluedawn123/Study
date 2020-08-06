import tensorflow as tf
import numpy as np
from keras.datasets import cifar100, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("x_train.shape : ", x_train.shape)  #(50000, 32, 32, 3)
print("x_test.shape : ", x_test.shape)    #(10000, 32, 32, 3)
print("y_train.shape : ", y_train.shape)  #(50000, 1)
print("y_test.shape : ", y_test.shape)    #(50000, 1)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")



#데이터 전처리 1. y의 원핫인코딩 #다중분류
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train의 shape : ", y_train.shape)   #(50000, 10)  


print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

#데이터전처리 2. x의 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255.                                                                                                                                     
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255.

print("아래는 Reshape 이후 x_test, x_train의 쉐이프이다.")
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)


input1 = Input(shape = (32, 32, 3) ) 
x = Conv2D(64, 3, activation='relu', padding="same")(input1)
x = Conv2D(64, 3, activation='relu', padding="same")(x)
x1 = MaxPooling2D(pool_size = 2)(x)
x2 = Conv2D(128, 3, activation='relu', padding="same")(x1)
x3 = Conv2D(128, 3, activation='relu', padding="same")(x2)
x4 = MaxPooling2D(pool_size = 2)(x3)
x5 = Conv2D(256, 3, activation='relu', padding="same")(x4)
x = Conv2D(256, 3, activation='relu', padding="same")(x5)
#x = layers.MaxPooling2D(2)(x)

x1 = Flatten()(x)
x = Dense(256, activation = 'relu')(x1)
x = Dense(256, activation = 'relu')(x)
x = Dense(100, activation='softmax')(x)
output1 = x

model = Model(inputs = input1, outputs = output1)
model.summary()

#3. 훈련                     
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
model.fit(x_train, y_train, epochs= 10, batch_size= 32, verbose = 2,
                 validation_split=0.2
                  )



#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss: ', loss)
print('acc: ', acc)

