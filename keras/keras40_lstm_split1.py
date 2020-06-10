import numpy as np
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

#1.데이터
a =  np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        aaa.append([item for item in subset])
    
    return np.array(aaa)

dataset = split_x(a, size)
print("=====================")
print(dataset)
print(dataset.shape)
print(type(dataset))


x = dataset[:, 0:4] #: 은 all, 즉 모든 행. 그리고 0:4는 0부터 4, 즉, 앞에 4개가 들어간다. 
y = dataset[:, 4]   #:은 역시 all, 그리고 4는 마지막 5번째만 가져오겠다는 의미.



print(x)
print(y)


print('x.shape : ', x.shape)              
print('y.shape : ', y.shape)  
'''
x = np.reshape(x, (6, 4, 1))       #x = x.reshape(x.shape[0], x.shape[1], 1)  과 같다.  여기의 배치사이즈는 총 행의 수. 그리고 아래의 배치사이즈는 그 행을 어떻게 나눌것이냐. 
                                   #x =  #위의 batch_size는 총 배치사이즈, 여기의 배치사이즈는 그곳에서 한 개씩 가져오겠다. 33줄 참조
print(x.shape)  

print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
#2. 모델구성
model = Sequential()
model.add(LSTM(10, input_shape = (4,1)))
model.add(Dense(6))
model.add(Dense(1))

from keras.callbacks import EarlyStopping



#3. 실행

early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 20)

model.compile(loss ='mse', optimizer = 'adam', metrics=['mse'] )
model.fit(x, y, epochs= 30, batch_size=1, verbose=1, callbacks=[early])

loss, mse = model.evaluate(x, y, batch_size = 1)
print("loss", loss)
print("mse", mse)

y_predict = model.predict(x)
print("y예측값:",y_predict)
'''