
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler      # (x - 최소) / (최대 - 최소)
from keras.layers.merge import concatenate, Concatenate 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

'''
def split_xy5(dataset, time_steps, y_column):  #(전체데이터, (행의수)뽑을 날짜의 갯수, 미래의 날짜의 갯수와 데이터값)
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]  #다음날(미래의것)의 몇 행으로 할래?(0,1,2,3,4)
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(hite, 4, 3)  # 중앙값 : 몇일로 자를래?   미래의 데이터를 몇개 뽑을래?예상할래?
print(x[0,:], "\n", y[0])
print(x.shape)
print(y.shape)
'''


'''
#아래 함수를 통해서 x1, x2를 분리 가능

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

'''
#keras40
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        aaa.append([j for j in subset])
    return np.array(aaa)

size = 6


#dataset = split_x(a, size)
#print("=====================")
#print(dataset)
#print(dataset.shape)
#print(type(dataset))


#npy불러오기
samsung = np.load('./data/samsung1.npy', allow_pickle=True)
hite = np.load('./data/hite1.npy', allow_pickle=True)

print(samsung)
print(hite)
print("samsung의 형태 : ", samsung.shape)   
print("hite의 형태 : ", hite.shape) 

samsung = samsung.reshape(samsung.shape[0], )

samsung = (split_x(samsung, size))
print(samsung.shape)                 #(504, 6)

x_sam = samsung[ : , 0:5]
y_sam = samsung[ : , 5]

print(x_sam.shape)       #    (504, 5)
print(y_sam.shape)        #    (504,)

x_hit = hite[5:510, : ]
print(x_hit.shape)         #    (504, 5)                  




# 2. 모델구성
input1 = Input(shape = (5, ))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape = (5, ))
x2 = Dense(5)(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1, x2])

output = Dense(10)(merge)

model = Model(inputs = [input1, input2],
              outputs = output)

model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x_sam, x_hit], y_sam, epochs = 5)








































