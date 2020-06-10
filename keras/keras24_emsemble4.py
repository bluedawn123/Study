#1. 데이터  1개가 들어가서 2개로 나오는 것. 
import numpy as np
x1 = np.array([range(1, 101), range(301,401)])    #리스트 아님  


y1 = np.array([range(711,811), range(611,711)])    
y2 = np.array([range(101,201), range(411,511)])
#print(x1.shape) #  열우선, 행무시

x1 = np.transpose(x1)  

y1 = np.transpose(y1)
y2 = np.transpose(y2)

#print(x1.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split
y2_train, y2_test = train_test_split(
     y2, shuffle=False, train_size=0.8)                                        ###위 두개랑 x1_train, x1_test, y1_train, y1_test, y2_test, y2_train = t.t.s(
                                                                               ###x1, y1, y2, shuffle=False, train_size = 0.8) 과 같다


#2. 모델구성. 모델 2개를 만들거니깐 shape가 2개가 필요하다. sequential로 불가능. 
from keras.models import Sequential, Model #함수형모델을 쓰겠다.  
from keras.layers import Dense, Input #인풋명시를 해야한다. 
#model = Sequential()

input1 = Input(shape=(2, )) 
dense1_1 = Dense(5, activation='relu')(input1) #꼬리에 모델명을 붙여준다. #1번째 히든레이어 완성. #input을 명시해야한다. 여기서는 input1
dense1_2 = Dense(4)(dense1_1)
dense1_3 = Dense(7)(dense1_2)


output1 = Dense(15)(dense1_3)
output1_2 = Dense(6)(output1)
output1_3 = Dense(12)(output1_2)
output1_4 = Dense(5)(output1_3)
output1_5 = Dense(2)(output1_4) #2로 나간다. 

output1 = Dense(30)(dense1_3)
output2_2 = Dense(7)(output1)
output2_3 = Dense(12)(output2_2)
output2_4 = Dense(7)(output2_3)
output2_5 = Dense(2)(output2_4) #2로 나간다. 

model = Model(inputs = input1, outputs= [output1_5, output2_5]) 

model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train, [y1_train, y2_train], epochs=30, batch_size=1, validation_split=0.25) #의미...?



#4. 평가, 예측
loss1, loss2, loss3, mse1, mse2 = model.evaluate(x1_test,
                        [y1_test, y2_test], batch_size=1)   


print("loss : ", loss1)
print("loss : ", loss2)
print("loss : ", loss3)
print("mse : ", mse2)
print("mse : ", mse1)

y1_predict, y2_predict = model.predict(x1_test)
print("y1 예측값 : ", y1_predict)
print("y2 예측값 : ", y2_predict)




# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))        
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE: ", (RMSE1 + RMSE2)/2)



# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2_1: ", r2_1)
print("R2_2: ", r2_2)
print("R2: ", (r2_1 + r2_2)/2)


