#과제 3 시퀀셜#제일 하단에 주석으로 acc와 loss 결과 명시 

import numpy as np
from keras.datasets import fashion_mnist, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")



#데이터 전처리 1. y의 원핫인코딩 #다중분류
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("원핫인코딩 후의 y_train의 shape : ", y_train.shape)   #(60000, 10)  

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

#데이터전처리 2. x의 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.                                                                                                                                     
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

print("아래는 Reshape 이후 x_test, x_train의 쉐이프이다.")
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print(" ")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")




#2. 모델 구성                               ☆☆☆☆와꾸 맞추김☆☆☆☆  dnn 이므로 (___ , )
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (28*28,  )))    #input_shape = (28 * 28, ) 로 대신해도 괜찮다. 
model.add(Dense(30, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(18, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'] )
model.fit(x_train, y_train, epochs = 100, batch_size = 32, shuffle = True,
                            validation_split =0.2)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size= 32)
print('loss: ', loss)
print('acc: ', acc)

#loss : 0.5480 acc : 0.8212