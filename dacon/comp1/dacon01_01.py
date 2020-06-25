import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import missingno as msno

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col=0)



x_test1 = test
y_predict = submission


print("train.shape : ", train.shape)                 #(10000, 75)      여기서 x_train x_test를 만들어야한다.
print("test.shape : ", test.shape)                   #(10000, 71)      test로 x_pred를 만들어야 한다. 
print("submission.shape : ", submission.shape)       #(10000, 4)                y_pred

#결측치
print(train.isnull().sum())  #트레인에서 isnull이 있는 것을 합해서 보여줘
train = train.interpolate()  #선형 보간법.
print(train.isnull().sum())
test = test.interpolate()

#train = train.fillna(method = 'bfill')  #뒤에 있는 거 끌어오기
#test = train.fillna(method = 'bfill')  #뒤에 있는 거 끌어오기

#numpy 저장.
train = train.values

np.save('./data/dacon/train.npy', arr=train)  

print("  ")
print("---------이후는 넘파이 변경 후 load해서 이어서 -------------")

train = np.load('./data/dacon/train.npy', allow_pickle=True)


#데이터
x = train[:, :71]     #[:]모든행  [0:72] 0~71까지의 인데스 
y = train[:, 71: ]    #[:]모든행 ,[71:] 71부터 까지의 인데스 

print("슬라이싱 후의 x의 형태 : ", x)
print("슬라이싱 후의 y의 형태 : ", y)

print("x.shape : ", x.shape)  #(10000, 71)
print("y.shape : ", y.shape)  #(10000, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.8)

print("x_train 모양 : ", x_train.shape)        #(8000, 71)         
print("x_test 모양 : ", x_test.shape)          #(2000, 71)     
print("y_train 모양 : ", y_train.shape)        #(8000, 4)       
print("y_test 모양 : ", y_test.shape)          #(2000, 4)



#2.모델
model = Sequential()
model.add(Dense(10, input_shape = (71, ), activation = 'relu'))  #들어가는게 71
model.add(Dense(23, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(35, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(27, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(4))  #y값에서 나가는 게 4



#3. 훈련
model.compile(loss = 'mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=3, batch_size=32)
# mse 는 회귀 acc는 분류 회귀는 1차함수 분류는 예측값의 범위가 정해져 있다.


#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("mae : ", mae)


y_pred = model.predict(x_test1)
print(y_pred)
'''
a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission3.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

'''
'''
# MAE 구하기
from sklearn.metrics import mean_absolute_error
def RMAE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
print("RMAE: ", RMAE(y_test, y_predict))


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2: ", r2_y_predict)

'''