from keras.datasets import cifar100, mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Flatten, Dropout, LSTM, Input
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler      # (x - 최소) / (최대 - 최소)
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')

modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'     # d : decimal, f : float
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')


print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")




#데이터 전처리 1. y의 원핫인코딩 #다중분류
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("원핫인코딩 후의 y_train의 shape : ", y_train.shape)   #(50000, 100)  


print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")





#데이터전처리 2. x의 정규화
x_train = x_train.reshape(50000, 32 * 32, 3).astype('float32') / 255.                                                                                                                                     
x_test = x_test.reshape(10000, 32 * 32, 3).astype('float32') / 255.

print("아래는 Reshape 이후 x_test, x_train의 쉐이프이다.")
print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)

print("")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")






####2.모델링 (LSTM의 함수형)

input1 = Input(shape= (32 * 32 , 3)) #함수형 모델이므로 2차원이다. x의 데이터 전처리에서 행무시이므로 앞에 2개가 들어간다. 
dense1 = LSTM(10)(input1) 
dense2 = Dense(13, activation='relu')(dense1)
dense3 = Dense(22, activation='relu')(dense2)
dense4 = Dense(13, activation='relu')(dense3)
dense5 = Dense(8, activation='relu')(dense4)

output1 = Dense(20)(dense5)
output2 = Dense(20)(output1)
output3 = Dense(100, activation='softmax')(output2)


model = Model(inputs = input1, outputs=output3) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

model.summary() #함수형모델


#3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 100, validation_split = 0.1, callbacks = [es, cp])






# 평가 및 예측
loss_acc = model.evaluate(x_test, y_test)
# print("res : ", res)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']


print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss_acc : ', loss_acc)


import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))               # 그래프의 크기를 (10, 6) 인치로

plt.subplot(2, 1, 1)                        # 2행 1열의 그래프 중 첫번째 그래프
'''x축은 epoch로 자동 인식하기 때문에 y값만 넣어준다.'''
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')              
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()                                  # 바탕에 격자무늬 추가
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)                        # 2행 1열의 두번째 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.grid()                                  # 바탕에 격자무늬 추가
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()