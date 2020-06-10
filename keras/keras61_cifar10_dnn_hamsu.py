

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

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")



#데이터 전처리 1. y의 원핫인코딩 #다중분류
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("원핫인코딩 후 y_train의 shape : ", y_train.shape)   #(50000, 10)  


print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")




#데이터전처리 2. x의 정규화 : DNN을 쓰려면 2차원이 필요하므로 reshape를 해야한다. 하지만!! 가장 앞은 행 무시이므로 총 3개가 필요하다. 
x_train = x_train.reshape(50000, 32 * 32 * 3,    ).astype('float32') / 255.                                                                                                                                     
x_test = x_test.reshape(10000, 32* 32 * 3,    ).astype('float32') / 255.

print("아래는 Reshape 이후 x_test, x_train의 쉐이프이다.")
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)



'''
정현이가 한 Keras57번 참조
# x_data 전처리 : Dense형 모델 사용을 위한 '2차원' reshape
x_train = x_train.reshape(60000, 28 * 28, ).astype('float32') / 255.
x_test = x_test.reshape(10000, 28 * 28, ).astype('float32') / 255.
print(x_train.shape)                  # (60000, 784)
print(x_test.shape)                  # (10000, 784)

'''


print("  ")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
####2.모델링


####2.모델링

input1 = Input(shape=(32 * 32 * 3,      )) #함수형 모델이므로 2차원이다. x의 데이터 전처리에서 행무시이므로 앞에 2개가 들어간다. 
dense1 = Dense(5, activation='relu')(input1) 
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(7, activation='relu')(dense2)
dense4 = Dense(5, activation='relu')(dense3)
dense5 = Dense(3, activation='relu')(dense4)

output1 = Dense(300)(dense5)
output2 = Dense(300)(output1)
output3 = Dense(10, activation='softmax')(output2)


model = Model(inputs = input1, outputs=output3) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

model.summary() #함수형모델


#3. 훈련                     
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
model.fit(x_train, y_train, epochs= 100, batch_size= 32, verbose = 2,
                 validation_split=0.2
                  )






#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 32)
print('loss: ', loss)
print('acc: ', acc)
































