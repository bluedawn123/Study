import numpy as np
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

#1.데이터
a =  np.array(range(1, 101))
size = 5                #time_steps = 4

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        aaa.append([item for item in subset])
    
    return np.array(aaa)

dataset = split_x(a, size)               #(96, 5)
print("=====================")
print(dataset)
print(dataset.shape)

print(type(dataset))

x = dataset[:90, 0:4] #: 은 all, 즉 모든 행. 그리고 0:4는 0부터 4, 즉, 앞에 4개가 들어간다. 
y = dataset[:90, 4]   #:은 역시 all, 그리고 4는 마지막 5번째만 가져오겠다는 의미.

x_predict = dataset[90:, 0:4]

print(x)
print(y)
print("x_predict의 형태 : ", x_predict)

print('x.shape : ', x.shape)              
print('y.shape : ', y.shape)  
print('x_predict.shape : ', x_predict.shape)

#실습 1. train test 분리할것.
#실습 2. 제일 마지막 x의 6행으로 예측하고 싶다. >>> 90행을 분리해야 한다. 비율은 8 : 2
#실습 3. validation을 넣을 것(train의 20%)

x = x.reshape(x.shape[0], x.shape[1], 1)    
                                   #x =  #위의 batch_size는 총 배치사이즈, 여기의 배치사이즈는 그곳에서 한 개씩 가져오겠다. 33줄 참조
print('x의 리쉐이프 형태 : ', x.shape)  

x_predict = x_predict.reshape(6, 4, 1)

print('x_predict의  리쉐이프 형태 : ', x.shape)  


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, shuffle=False)    # 일단 6:4


#2. 모델구성

model = Sequential()
model.add(LSTM(10, input_shape = (4,1)))
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

from keras.callbacks import EarlyStopping



early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 20)


#3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=20, batch_size=1, validation_split=0.2, verbose=1, shuffle = True)

#4. 평가
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
#loss, mse = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.
print("loss : ", loss)
print("mse : ", mse)
print("loss", loss)
print("mse", mse)

y_predict = model.predict(x_test)
print("y예측값:",y_predict)
