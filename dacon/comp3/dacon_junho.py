import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_features = pd.read_csv('./data/dacon/comp2/train_features.csv', header = 0, index_col=0)
test_features = pd.read_csv('./data/dacon/comp2/test_features.csv', header = 0, index_col=0)
submission = pd.read_csv('./data/dacon/comp2/sample_submission.csv', header = 0, index_col=0)
train_target = pd.read_csv('./data/dacon/comp2/train_target.csv', header = 0, index_col=0)


print("train_features.shape : ", train_features.shape)  #(1050000, 5)      
print("test_features.shape : ", test_features.shape)    #(262500, 5)     
print("train_target.shape : ", train_target.shape)      #(2800, 4)         
print("submission.shape : ", submission.shape)          #(700, 4)               

#결측값 확인
print(train_features.isnull().sum())  #결측값 x
print(train_target.isnull().sum())    #결측값 x
print(test_features.isnull().sum())   #결측값 x


#넘파이 변경하기
train_features = train_features.values
train_target = train_target.values
submission = submission.values
test_features = test_features.values

x1 = train_features[0:375, 1:]
y1 = train_target[0:375, : ]

print("x1.shape : ", x1.shape)    # (375, 4)  #lstm 써주려고 4, 1 로 변경한다. 
x1 = x1.reshape(x1.shape[0], 4, 1)


print("reshape 이후 x1.shape : ", x1.shape)    # (375, 4, 1)
print("y1.shape : ", y1.shape)    # (375, 4) 

#y1 = y1.reshape(y1.shape[0], 4, 1)
#print("y1.shape : ", y1.shape)    # (375, 4, 1) 필요없음;


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x1, y1, shuffle=False, train_size=0.8)


model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(4,1)))  #인풋쉐이프가 3,1의 의미?? >>  #4,3,1에서 4무시. (3,1)을 모델의 기준으로 잡겠다. 
model.add(Dense(15))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(4))  #나가는게4
model.summary()

#컴파일, fit, 예측
model.compile(optimizer='adam', loss = 'mse',  metrics=['mse'])
model.fit(x_train, y_train, epochs =10, batch_size = 12 #,callbacks = [es] 
          )                

loss, mse = model.evaluate(x_train, y_train, batch_size = 1)
print("loss", loss)
print("mse", mse)

y_predict = model.predict(x_test)
print("y예측값:",y_predict)

