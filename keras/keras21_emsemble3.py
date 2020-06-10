#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311,411), range(100)])     
y1 = np.array([range(711,811), range(711,811), range(100)])    

x2 = np.array([range(101,201), range(411,511), range(100, 200)])
y2 = np.array([range(501,601), range(711,811), range(100)])
print(x1.shape) #  열우선, 행무시

x1 = np.transpose(x1)  #100행 3열.
y1 = np.transpose(y1)
y2 = np.transpose(y2)
x2 = np.transpose(x2)

print(x1.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=False, train_size=0.8)


#2. 모델구성. 모델 2개를 만들거니깐 shape가 2개가 필요하다. sequential로 불가능. 
from keras.models import Sequential, Model #함수형모델을 쓰겠다.  
from keras.layers import Dense, Input #인풋명시를 해야한다. 

#model = Sequential()
#model.add(Dense(5, input_dim = 3))
#model.add(Dense(4))
#model.add(Dense(1))
#함수형 모델로 변경
#인풋, 아웃풋이 뭔지 명시해야한다.

#모델1
input1 = Input(shape=(3, )) #인풋 3
dense1_1 = Dense(5, activation='relu', name='bitking1')(input1) #꼬리에 모델명을 붙여준다. #1번째 히든레이어 완성. #input을 명시해야한다. 여기서는 input1
dense1_2 = Dense(4, activation='relu', name='bitking2')(dense1_1)
dense1_3 = Dense(7, activation='relu', name='bitking3')(dense1_2)


#모델2
input2 = Input(shape=(3, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다. #함수형 모델은 기본적으로 input shape 로 한다. , 레이어는 이름을 지정해줘야하므로 input1으로 지정
dense2_1 = Dense(2, activation='relu')(input2) #꼬리에 모델명을 붙여준다. #1번째 히든레이어 완성. #input을 명시해야한다. 여기서는 input1
dense2_2 = Dense(6, activation='relu')(dense2_1)
dense2_3 = Dense(3, activation='relu')(dense2_2)


from keras.layers.merge import concatenate #단순병합
merge1 = concatenate([dense1_3, dense2_3]) #2개 이상은list [] 사용...

middle1 = Dense(30)(merge1)  #노드 30개 생성
middle2 = Dense(5)(middle1)
middle3 = Dense(4)(middle2) 

#아웃풋 구성해야한다. 이제.
#3개짜리 모델2개이다. 아웃풋도 2개이므로, 이것을 만들어줘야한다. 
#output 모델구성

output1 = Dense(25)(middle3)
output1_2 = Dense(4)(output1)
output1_3 = Dense(3)(output1_2) #3으로 나간다. 

output1 = Dense(30)(middle3)
output2_2 = Dense(7)(output1)
output2_3 = Dense(3)(output2_2) #3으로 나간다. 



model = Model(inputs = [input1, input2], outputs= [output1_3, output2_3]) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시) #list로 구성. 

model.summary() #함수형모델

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train], 
          epochs=25, batch_size=1, 
          validation_split=(0.2), verbose=1) #verbose를 사용하여 머신의 훈련 을 보이지않게 생략하여 빠른 훈련이 가능하다

#4. 평가, 예측
loss1 = model.evaluate([x1_test, x2_test],
                       [y1_test, y2_test], 
                           batch_size=1)   #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss1)
#이 경우 loss, mse의 의미...? emsemble2.py와는 왜 다르게 이렇게 함?

print(model.metrics_names)
#y1, y2를 나눠서 이렇게하는 이유? 나가는게 2가지 이기 때문...?
y1_predict, y2_predict = model.predict([x1_test, x2_test])
'''
print("=================================")
print(y1_predict)
print("=================================")
print(y2_predict)


# RMSE 구하기. 여기서도 y가 2가지 이기 때문에 이렇게...?
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))        

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE: ", (RMSE1 + RMSE2)/2)

# R2 구하기 (여기서도 y가 2가지 이기 때문에 이렇게...?)
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)

print("R2_1: ", r2_1)
print("R2_2: ", r2_2)
print("R2_2: ", (r2_1 + r2_2)/2)

#100행3열 짜리 2개가 들어가서 100행3열 짜리가 2개가 나왔다. 
# 가장 많이 나오는 형태는, 여러개를 넣고 하나가 나오는 경우이다. 
'''