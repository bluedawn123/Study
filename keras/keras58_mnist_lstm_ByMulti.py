import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, LSTM
from keras.models import Sequential




from keras.datasets import mnist  #keras.dataset => mnist에서 예제파일 소환

(x_train, y_train), (x_test, y_test) = mnist.load_data()  #mnist에 분류가 되어있다.


print("x_train의 모양 ", x_train.shape)      # (60000, 28, 28)
print("x_test의 모양 ", x_test.shape)        # (10000, 28, 28) 
print("y_train의 모양 ", y_train.shape)      # 1디멘션 (60000,)
print("y_test의 모양 ", y_test.shape)        #(10000,)


#데이터 전처리 1. y의 원핫인코딩 #다중분류
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train의 shape : ", y_train.shape)   #(6만, 10)  여기서 10으로 해주는 이유?!?!?!?




#데이터전처리 2. x의 정규화
x_train = x_train.reshape(60000, 28 * 28, 1).astype('float32') / 255.    #minmax scalar
                                                                      #최소로 만들어주려고 하는 건 알겠는데 왜 255로 나누는 이유...? 왜 255임..?
                                                                      #색이 255로 구분되기 때문.  0부터 255가 들어가 있다. 그러므로 최소~최대가 0~255때문
x_test = x_test.reshape(10000, 28 * 28, 1).astype('float32') / 255.

#위의 의미 : 1. reshape로 4차원 만듦 : cnn모델이 input_shape가 batchsize, 세로, 가로, 채널 이므로.
#           2. 0실수형으로 변경. 
#           3. 255로 나눔. 0부터 1까지의 사이에 넣기 위해. 즉 정규화(minmax)
#☆☆☆왜 255를 나누는지 모를떄는 minmax를 사용하면 된다.





###2. 모델링
model = Sequential()
model.add(LSTM(10, input_shape = (28 * 28, 1)))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(10,activation='softmax'))
model.summary()



'''
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
'''








