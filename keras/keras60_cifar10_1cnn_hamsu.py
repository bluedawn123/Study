

import numpy as np
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

#plt.imshow(x_train[3])
#plt.show()

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

print("")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
####2.모델링

input1 = Input(shape = (32, 32, 3) )   #CNN을 사용했으므로 4차원
dense1 = Conv2D(200, (3, 3), padding = 'same')(input1)
maxpool1 = MaxPooling2D(pool_size=2)(input1)
drop1 = Dropout(0.2)(maxpool1)                                             # Dropout 사용

dense2 = Conv2D(100, (2, 2), padding = 'same')(drop1)
maxpool2 = MaxPooling2D(pool_size=2)(dense2)
drop2 = Dropout(0.3)(maxpool2)                                            # Dropout 사용

dense3 = Conv2D(80, (2, 2), padding = 'same')(drop2)
maxpool3 = MaxPooling2D(pool_size=2)(dense3)
drop3 =Dropout(0.3)(maxpool3)                                             # Dropout 사용

dense4 = Conv2D(60, (2, 2),padding = 'same')(drop3)
                                        # Dropout 사용

dense5 = Conv2D(40, (2, 2),padding = 'same')(dense4)
drop5 = Dropout(0.3)(dense5)                                            # Dropout 사용

dense6 = Conv2D(20, (2, 2), padding='same')(drop5)
flat = Flatten()(dense6)
output1 = Dense(10, activation='softmax')(flat)              

model = Model(inputs = input1 , outputs = output1)

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
