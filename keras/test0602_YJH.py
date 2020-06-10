
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler      # (x - 최소) / (최대 - 최소)
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'

cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto', save_weights_only = False, verbose = 1)
'''


hite = pd.read_csv('./data/csv/hite.csv', index_col = 0, header=0, encoding='cp949', sep=',') 
samsung = pd.read_csv('./data/csv/samsung.csv', index_col = 0, header=0, encoding='cp949', sep=',') 
  
samsung = samsung.dropna()#삼성 nan 값 제거
hite = hite.dropna() #하이트 nan 제거

print(hite)
print(samsung) 
print("hite의 형태 : ", hite)  #(508, 5)          ☆☆☆☆맞지않는다
print("samsung의 형태 : ", samsung)   #(509, 1)                    "




#하이트의 모든 데이터를 str -> int형
for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',', ''))
#하이트를 오름차순 형으로
hite = hite.sort_values(['일자'], ascending = [True])
print(hite)



#삼성의 모든 데이터를 str -> int형
for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',', ''))
#삼성전자 를 오름차순 형으로
samsung = samsung.sort_values(['일자'], ascending=[True])
print(samsung)



print("  ")
print("------------------이하 넘파이 변경 후-----------------------")
print("중요한 것은 넘파이 변경 후 우리가 수업대로 쓸 수 있도록 변경 된다는 것이다.")
#pandas를 numpy로 변경 후 저장
samsung = samsung.values
hite = hite.values

print(type(samsung), type(hite))  #넘파이로 저장 완료
print(samsung.shape, hite.shape)  #삼성은 (509, 1) 하이트는 (508,5) 후에 와꾸 변경을 해줘야 한다. 

np.save('./data/samsung.npy', arr=samsung)  #굳이 할 필요 없지만 책에 나온거라 한번 해봤다. 
np.save('./data/hite.npy', arr=hite)


print("  ")
print("---------이후는 넘파이 변경 후 load해서 이어서 -------------")

hite = np.load('./data/hite.npy', allow_pickle=True)
samsung = np.load('./data/samsung.npy', allow_pickle=True)





samsung = samsung[:-1, :]  #마지막 6.2일치를 잘라버림 ....#삼성전자의6월 2일치 데이터를 제거한다.
print(samsung.shape)   #(508, 1)


'''


print("----------------------------------------------")
print("삼성과 하이트의 모양 완성")

   
  
print("samsung의 형태 : ", samsung.shape)  #(508, 1)
print("hite의 형태  : ", hite.shape)       #(508, 5)


#앙상블을 만들어 주기 위해서 X1, X2, Y1, Y2 설정

print(" ")
print("----------------------------------------------")
print(" ")


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0] #몇 행으로 할래? 삼성은 하나의 열이므로 이건 항상 0으로 고정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 3, 1)         #(4, 1)이면 4일치로 다음날 것을 예측하겠다. 
x2, y2 = split_xy5(hite, 3, 1)            #(3, 1)이면 3일치로 다음 날을 예측하겠다. 

print(x2[0,:],"\n", y2[0])                #하이트의 미래 시가 예측
print("하이트 예측의 형태 : ", x2.shape)

print(x1[0,:],"\n", y1[0])                #삼성의 미래 시가 예측
print("삼성 예측의 형태 : ", x1.shape)     # (505, 3, 1)

print("y2의 형태 : ", y2.shape)    #    (505, 1)
print("y1의 형태 : ", y1.shape)     #   (505, 1)
print("x2의 형태 : ", x2.shape)      #  (505, 3, 5)
print("x1의 형태 : ", x1.shape)     #  (505, 3, 1)

print("----------------------------------------------")
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=1, test_size = 0.3)



print("x1_train의 형태 : ", x1_train.shape)          #  (353, 3, 1)
print("x1_test의 형태 : ", x1_test.shape)            #  (152, 3, 1)
print("x2_train의 형태 : ", x2_train.shape)          #  (353, 3, 5)
print("x2_test의 형태 : ", x2_test.shape)            #  (152, 3, 5)
print("y1_train의 형태 : ", y1_train.shape)          #  (353, 1)
print("y1_test의 형태 : ", y1_test.shape)            #  (152, 1)
print("y2_train의 형태 : ", y2_train.shape)          #  (353, 1)
print("y2_test의 형태 : ", y2_test.shape)            #  (152, 1)

print("------------------리쉐이프 후 ------------------")
print(" ")

x1_train = np.reshape(x1_train,
    (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,
    (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,
    (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape)      #(353, 15)
print(x2_test.shape)       #(152, 15)


print("x1_train의 형태 : ", x1_train.shape)          #  (353, 3)
print("x1_test의 형태 : ", x1_test.shape)            #  (152, 3)
print("x2_train의 형태 : ", x2_train.shape)          #  (353, 15)
print("x2_test의 형태 : ", x2_test.shape)            #  (152, 15)


#### 데이터 전처리 #####
scaler1 = StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)


scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x2_train_scaled[0, :])
print(x1_train_scaled[0, :])
print(x2_test_scaled[0, :])
print(x1_test_scaled[0, :])


print("------------------모델링------------------")

# 3. 모델구성
input1 = Input(shape=(3, ))
dense1 = Dense(64)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(15, ))
dense2 = Dense(64)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
middle1 = Dense(12, activation = 'relu')(merge)
middle2 = Dense(32, activation = 'relu')(middle1)
middle3 = Dense(10, activation = 'relu')(middle2)

output4 = Dense(16, activation = 'relu')(middle3)
output5 = Dense(1, activation = 'relu')(output4)


model = Model(inputs=[input1, input2],
              outputs = output5 )


model.summary()


print("------------------예측 ------------------")

#4. 예측
'''
'''
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit([x1_train_scaled, x2_train_scaled], y1_train, 
          verbose=1, batch_size=1, epochs=10 )

loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)
'''
'''
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=13)
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, 
          verbose=1, batch_size=1, epochs=30, 
          callbacks=[early_stopping])

loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)


#스케일된 삼성,하이트(test)의 값으로 y를 예측한다. 이미 loss와 mse를 안다. 

y1_pred = model.predict([x1_test_scaled, x2_test_scaled])

for i in range(10):  #10일치 계산하는 경우.
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])



# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict ):
    return np.sqrt(mean_squared_error(y1_test, y1_pred))
print("RMSE: ", RMSE(y1_test, y1_pred))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y1_pred)
print("R2: ", r2_y_predict)
'''