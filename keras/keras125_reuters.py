from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import keras

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 1000, test_split = 0.2) #가장많이 쓰는 단어 1000개

print("x_train.shape : ", x_train.shape)
print("x_test : ", x_test.shape)
print("y_test : ", y_test.shape)
print("y_train.shape : ", y_train.shape)

print("x_train[0] : ", x_train[0])  #가장 많이 나오는 단어.
print("y_train[0] : ", y_train[0])
#print("x_train[0].shape : ", x_train[0].shape )  #리스트형은 shape불가

print(len(x_train[0]))   #[0]은 87개의 단어
#크기가 다른 리스트를 크기와 간격을 갖게 해줘야한다. 빈자리는 0으로 채우고. 

#y의 카테고리 갯수 확인
category = np.max(y_train) + 1   #+1은 인덱스가 0부터 시작이니깐.
print("카테고리 : ", category)    #46개. 0부터 45까지 있다는 것. 

#y의 유니크한 값들 출력. 
y_bunpo = np.unique(y_train)
print("y_bunpo : ", y_bunpo)    #0부터 45까지 분포되어 있다는 걸 알 수 있다. 총 46개 존재!!

#y의 분포확인
y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()  
print("갯수 확인 : ", bbb)

#groupby 사용법 숙지할
from keras.preprocessing.sequence import pad_sequences  
from keras.utils.np_utils import to_categorical

'''
x_train = pad_sequences(x_train, maxlen=100, padding = 'pre' )    #maxlen = n . 최대값을 n개로 잡는다
                                                                    #truncating = 잘리는 경우, 앞에서 잘라서 삭제하겠다.

print(len(x_train[0]))
print(len(x_train[-1]))

y_train = to_categorical(y_train)   
y_test = to_categorical(y_test)     

print("y_train.shape : ", y_train.shape)  #(8982, 46)
print("y_test.shape : ", y_test.shape)    #(2246, 46)
'''




## pad_sequences 사용하여 데이터 shape 맞춰주기
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen = 100,
    padding = 'pre', dtype = 'int32')
x_test = keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen = 100,
    padding = 'pre', dtype = 'int32')
print(len(x_train[0]))      # 100
print(len(x_train[-1]))     # 100
y_train = to_categorical(y_train)   
y_test = to_categorical(y_test)     

print("y_train.shape : ", y_train.shape)  #(8982, 46)
print("y_test.shape : ", y_test.shape)    #(2246, 46)


#모델
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten

model = Sequential()
#model.add(Embedding(1000, 100, input_length=100))
model.add(Embedding(1000, 128))
model.add(LSTM(100))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])
history = model.fit(x_train, y_train, batch_size=100, epochs = 3, validation_split=0.2)
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker ='.', c ='red', label ='TestSet Loss')
plt.plot(y_loss, marker ='.', c ='blue', label ='TrainSet Loss')
plt.legend(loc ='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


