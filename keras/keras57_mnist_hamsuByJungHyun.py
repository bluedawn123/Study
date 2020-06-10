import numpy as np

#1. 데이터
from keras.datasets import mnist
mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                    # (60000, 28, 28)
print(x_test.shape)                    # (10000, )
print(y_train.shape)                    # (60000, )
print(y_test.shape)                    # (10000, )

print("x_train 모양 : ", x_train.shape)                 
print("x_test 모양 : ", x_test.shape)                  
print("y_train 모양 : ", y_train.shape)                   
print("y_test 모양 : ", y_test.shape) 


# x_data 전처리 : Dense형 모델 사용을 위한 '2차원' reshape
x_train = x_train.reshape(60000, 28 * 28, ).astype('float32') / 255.
x_test = x_test.reshape(10000, 28 * 28, ).astype('float32') / 255.
print("전처리 전 x_train의 shapel : ", x_train.shape)                  # (60000, 784)
print("전처리 전 x_test의 shape : ", x_test.shape)                  # (10000, 784)


# y_data 전처리 : one_hot_encoding (다중 분류)
from keras.utils.np_utils import to_categorical
#from keras.utils import np_utils 는 필요가 없는가?
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("원핫인코딩 후의 y_train의 shape : ", y_train.shape)             #60000, 10
print("원핫인코딩 후의 y_test의 shape : ", y_test.shape)               #10000, 10




#2. 모델 구성
from keras.models import Sequential, Model #함수형모델을 쓰겠다.
from keras.layers import Dense, Input

input1 = Input(shape=(784, ))                #함수형 모델이므로 2차원
dense1 = Dense(5, activation='relu')(input1) #꼬리에 모델명을 붙여준다. #1번째 히든레이어 완성. #input을 명시해야한다. 여기서는 input1
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
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'] )
model.fit(x_train, y_train, epochs = 100, batch_size = 32, shuffle = True,
                            validation_split =0.2)











#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size= 32)
print('loss: ', loss)
print('acc: ', acc)





