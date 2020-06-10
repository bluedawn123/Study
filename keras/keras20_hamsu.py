#1. 데이터

import numpy as np
x = np.array([range(1, 101), range(311,411), range(100)])
y = np.array(range(711,811)) 

print(x.shape) #  열우선, 행무시 (3,100 )

x = np.transpose(x)  #(100,3)
y = np.transpose(y)

print(x.shape) #(100,3)
print(y.shape) #(100, )
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.8)




print("x_train 모양 : ", x_train.shape)                 
print("x_test 모양 : ", x_test.shape)                  
print("y_train 모양 : ", y_train.shape)                   
print("y_test 모양 : ", y_test.shape) 









#2. 모델구성
from keras.models import Sequential, Model #함수형모델을 쓰겠다.
from keras.layers import Dense, Input #인풋명시를 해야한다. 

#model = Sequential()
#model.add(Dense(5, input_dim = 3))
#model.add(Dense(4))
#model.add(Dense(1))
#함수형 모델로 변경
#인풋, 아웃풋이 뭔지 명시해야한다.

input1 = Input(shape=(3, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다. #함수형 모델은 기본적으로 input shape 로 한다. , 레이어는 이름을 지정해줘야하므로 input1으로 지정
dense1 = Dense(5, activation='relu')(input1) #꼬리에 모델명을 붙여준다. #1번째 히든레이어 완성. #input을 명시해야한다. 여기서는 input1
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(7, activation='relu')(dense2)
dense4 = Dense(5, activation='relu')(dense3)
dense5 = Dense(3, activation='relu')(dense4)
output1 = Dense(1)(dense5)  #여기서 dense(1) 인 이유는..? 나가는 게 1개 이기 때문..!!

model = Model(inputs = input1, outputs=output1) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시)

model.summary() #함수형모델

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=25, batch_size=1, validation_split=(0.2), verbose=2) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("mse : ", mse)


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
