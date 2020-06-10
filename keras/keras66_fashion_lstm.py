#과제4 시퀀셜#제일 하단에 주석으로 acc와 loss 결과 명시 

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
x_train = x_train.reshape(60000, 28, 28).astype('float32') / 255.                                                                                                                                     
x_test = x_test.reshape(10000, 28, 28).astype('float32') / 255.

print("아래는 Reshape 이후 x_test, x_train의 쉐이프이다.")
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print(" ")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")


###2. 모델링  ☆☆☆☆와꾸 맞추김☆☆☆☆  lstm 이므로 (___ , ___)
model = Sequential()
model.add(LSTM(10, input_shape = (28, 28)))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(18))
model.add(Dense(32))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(10,activation='softmax'))
model.summary()

#3
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=5, batch_size = 2)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=2) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("acc : ", acc)


x_pred = (x_test)
y_predict = model.predict(x_pred)
print('y_pred :', y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

#