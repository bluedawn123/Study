
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, History, TensorBoard





#1. 데이터
a = np.array(range(1, 101))
size = 5                   # time_steps = 4

#LSTM  모델을 완성하시오.

def split_x(seq, size):
    aaa = [] 
    for i in range(len(seq) - size + 1):#seq - size +1  = 값 행종료값
        subset = seq[i: (i+size)] #열 지정
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)  # (6,5)
print(dataset)
print(dataset.shape)
print(type(dataset))



x = dataset[:90, 0:4] #[:]모든행 ,  [0:4] 0~3까지의 인데스 
y = dataset[:90, -1:]   # 4 인덱스 데이터를 가져온다.
x_predict = dataset[90:, 0:4]


#x = np.reshape(x, (90, 4, 1 ))
x = x.reshape(x.shape[0], x.shape[1], 1) 
x_predict = np.reshape(x_predict, (6, 4, 1 ))
#x = x.reshape(6, 4, 1)
print(x.shape)











#2. 모델
from keras.models import load_model
#model = load_model('./model/save_44.h5')

model = Sequential()
model.add(LSTM(10, input_shape = (4,1)))
model.add(Dense(6))
model.add(Dense(1))

model.summary()

#3. 훈련
tb_hist = TensorBoard(log_dir='graph', histogram_freq =0, write_graph=True, write_images=True)

early = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 15)

model.compile(loss ='mse', optimizer = 'adam', metrics=['acc'])
hist = model.fit(x, y, epochs= 50, batch_size=1, verbose=1, callbacks=[early, tb_hist], validation_split=0.2)

print(hist)
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc ')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()

'''
loss, mse = model.evaluate(x, y, batch_size = 1)
print("loss", loss)
print("mse", mse)

y_predict = model.predict(x_predict)
print("y예측값:",y_predict)
'''