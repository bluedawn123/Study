#1. 데이터  2개가 들어가서 하나로 나오는 것. 
import numpy as np
x1 = np.array([range(1, 101), range(311,411), range(100)])     
x2 = np.array([range(711,811), range(711,811), range(511,611)])    

y1 = np.array([range(101,201), range(411,511), range(100)])
#print(x1.shape) #  열우선, 행무시

x1 = np.transpose(x1)  #100행 3열.
y1 = np.transpose(y1)
x2 = np.transpose(x2)

#print(x1.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split
y1_train, y1_test = train_test_split(
    y1, shuffle=False, train_size=0.8)


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
dense1_1 = Dense(18, activation = 'relu')(input1)  # 첫번째 모델의 첫번째 히든레이어 구성
dense1_2 = Dense(17, activation = 'relu')(dense1_1)  # 첫번째 모델의 두번째 히든레이어 구성
dense1_3 = Dense(18, activation = 'relu')(dense1_2)
dense1_4 = Dense(21)(dense1_3)



#모델2
input2 = Input(shape = (3, ))
dense2_1 = Dense(13, activation = 'relu')(input2)
dense2_2 = Dense(26, activation = 'relu')(dense2_1)
dense2_3 = Dense(24, activation = 'relu')(dense2_2)
dense2_4 = Dense(18)(dense2_3)


from keras.layers.merge import concatenate #단순병합
merge1 = concatenate([dense1_4, dense2_4]) #2개 이상은list [] 사용...

middle1 = Dense(29)(merge1)
middle2 = Dense(33)(middle1)
middle3 = Dense(14)(middle2)

#아웃풋 구성해야한다. 이제.
#3개짜리 모델2개이다. 아웃풋도 2개이므로, 이것을 만들어줘야한다. 
#output 모델구성

output1 = Dense(31)(middle3)
output2 = Dense(37)(output1)
output3 = Dense(3)(output2) #3으로 나간다. 
 

model = Model(inputs = [input1, input2], outputs=output3) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시) #list로 구성. 

model.summary() #함수형모델

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          y1_train, 
          epochs=200, batch_size=1, 
          validation_split=(0.25), verbose=1)

#4. 평가, 예측
loss1 = model.evaluate([x1_test, x2_test],
                       [y1_test], 
                           batch_size=1)   


print("loss : " , loss1)

y1_predict = model.predict([x1_test, x2_test])


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))        

RMSE1 = RMSE(y1_test, y1_predict)
print("RMSE1 : ", RMSE1)


# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)

print("R2_1: ", r2_1)


#100행3열 짜리 2개가 들어가서 100행3열 짜리가 2개가 나왔다. 
# 가장 많이 나오는 형태는, 여러개를 넣고 하나가 나오는 경우이다. 
