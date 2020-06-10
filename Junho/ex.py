
# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Flatten
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np

# 1-1. 객체 생성
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
# scaler = MinMaxScaler()       # 최소/최대값이 0, 1이 되도록 스케일링
scaler = StandardScaler()     # 평균이 0이고 표준편차가 1인 정규분포가 되도록 스케일링

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [2000, 3000, 4000], [3000, 4000, 5000], [4000, 5000, 6000],
              [100, 200, 300]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400])
x_predict = np.array([[55, 65, 75]])



scaler.fit(x)   # 실행하다
x = scaler.transform(x)
x_predict = scaler.transform(x_predict)
print(x)
print(x_predict)



x = x.reshape(14, 3, 1)
x_predict = x_predict.reshape(1, 3, 1)



# 3. 모델 구성


# LSTM(return_sequences) _ 함수형 모델
input1 = Input(shape = (3, 1))
dense1 = LSTM(51, return_sequences = True)(input1)
dense2 = LSTM(48, return_sequences = True)(dense1)
dense3 = LSTM(49)(dense2)
dense4 = Dense(45)(dense3)
dense5 = Dense(45)(dense4)
dense6 = Dense(45)(dense5)
dense7 = Dense(45)(dense6)

output1 = Dense(31)(dense7)
output2 = Dense(16)(output1)
output3 = Dense(1)(output2)
output4 = Dense(1)(output3)
output5 = Dense(1)(output4)
output6 = Dense(1)(output5)

model = Model(inputs = input1,
              outputs = output6)


# 4. 실행
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x, y,
          epochs = 100, batch_size = 10,
          callbacks = [early])

# 5. 예측
y_predict = model.predict(x_predict)

print(x_predict)
print(y_predict)