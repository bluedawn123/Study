# 20-07-02_27
# 한 개의 모델에서 분류와 회귀를 동시에 나오게 할 수 있을까?


### 1. 데이터
import numpy as np
from sklearn.metrics import r2_score

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])


### 2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()
model.add(Dense(50, input_shape = (1, )))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1, activation='sigmoid'))      #0.5 이하는 0, 초과는 1로 배출하는 함수

model.summary()


### 3. 실행, 훈련
model.compile(loss = ['binary_crossentropy'], optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=1)


### 4. 평가, 예측
loss = model.evaluate(x_train, y_train)
print('loss :', loss)

x1_pred = np.array([11, 12, 13, 14])

y_pred = model.predict(x1_pred)
print("예측값들 : ", y_pred)


loss, acc = model.evaluate(x_train, y_train, batch_size= 32)
print("  ")
print('acc: ', acc)
