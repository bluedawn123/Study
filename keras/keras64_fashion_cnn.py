#과제 2
#제일 하단에 주석으로 acc와 loss 결과 명시 

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


###2. 모델링                 ☆☆☆☆와꾸 맞추김☆☆☆☆  Cnn에 시퀀셜 이므로 (___ , ___, ___)

model = Sequential()
model.add(Conv2D(10, (2,2),  input_shape=(28,28,1)))  #(2,2) = 픽셀을 2 by 2 씩 잘른다.
                                                      #(가로,세로,명암 1=흑백, 3=칼라)
                                                      #(행, 열 ,채널수) # batch_size, height, width, channels

model.add(Conv2D(7,(3,3)))    #strides : 높이와 너비를 따라 컨벌루션의 보폭을 지정하는 정수 또는 튜플 / 2 개의 정수 목록입니다. 모든 공간 치수에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다. 모든 보폭 값! = 1을 지정하면 모든 dilation_rate값! = 1 을 지정할 수 없습니다 .
model.add(Conv2D(5,(2,2)))
model.add(Conv2D(5,(2,2)))
model.add(Conv2D(5,(2,2), strides=2))
model.add(Conv2D(5,(2,2),strides=2, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten()) # 2차원으로 변경
model.add(Dense(10,activation='softmax'))
model.summary()


#훈련
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=5, batch_size = 2)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=2) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)

#loss : 0.4995234   acc : 0.83241211