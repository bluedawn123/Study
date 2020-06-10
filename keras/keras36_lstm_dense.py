#36. Lstm 모델을 dense 함수로


## 함수형으로 리뉴얼하시오.

# 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Flatten
from keras.callbacks import EarlyStopping
import numpy as np


#  EarlyStopping 객체 생성
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],            
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[11,12,13],
           [20,30,40],[30,40,50],[40,50,60],
          ])                                                       #(13, 1)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])                      # (13, )   벡터

print('x.shape : ', x.shape)               # (13, 3)
print('y.shape : ', y.shape)  



x_predict = np.array([55, 65, 75])           # (3, )
x_predict = x_predict.reshape(1, 3)

print("x 예측값의 차원 : ", x_predict.shape)

#print(x.shape)                            #LSTM에 넣기 위해서 (13, 3, 1)    

print("x 예측값의 차원 : ", x_predict.shape)

#2. 모델구성
''' #36. 이 부분을 Dense형으로 바꿔라!
이건 시퀀셜형model = Sequential()

#model.add(LSTM(10, activation = 'relu', input_shape=(3,1)))  #인풋쉐이프가 3,1의 의미?? >>  #4,3,1에서 4무시. (3,1)을 모델의 기준으로 잡겠다. 
model.add(LSTM(10, input_length=3, input_dim = 1, return_sequences=True))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(5))
model.add(Dense(1))   #☆☆☆☆☆☆☆질문 2. 1인 이유?!?!?!?!?☆☆☆☆☆☆☆

model.summary()'

이건 함수형
input1 = Input(shape=(3, 1))
dense1 = LSTM(10, activation='relu', input_shape = (3, 1), return_sequences =True)(input1)
dense1_1 = LSTM(10, return_sequences =True)(dense1)
dense2 = Dense(5)(dense1_1) 
dense3 = Dense(1)(dense2) 


output1 = Dense(100)(dense3)
output2 = Dense(100)(output1)
output3 = Dense(1)(output2)

'''

#2. 모델구성

model = Sequential()

model.add(Dense(30, activation = 'relu', input_shape = (3, )))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(1))

model.summary() 


# # EarlyStopping
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')

# 4. 실행
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x, y,
          epochs = 100, batch_size = 10,
          callbacks = [early])

# 5. 예측
y_predict = model.predict(x_predict)

print(x_predict)
print(y_predict)