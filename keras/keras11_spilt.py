#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201)) #w  는1 b는100

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = x[:60]
y_val = x[60:80]
y_test = x[80:]


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10, input_dim = 1))  #x는 60개가 한 덩이로 들어가기 때문에 가능
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1)) #한 덩이로 역시 나오기 때문에 '1'이 받을 수 있다.

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=30, batch_size=1, validation_data=(x_val, y_val))
# mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.


#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("mse : ", mse)

#y_pred = model.predict(x_pred)
#print("y_predict : ", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)