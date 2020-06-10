# 다중 분류
from sklearn.datasets import load_iris
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Flatten, Dropout, LSTM, Input
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 콜백 3
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
iris = load_iris()


#얼리스타핑
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose =1)
#텐서보드
ts_board = TensorBoard(log_dir = 'graph', histogram_freq= 0,
                      write_graph = True, write_images=True)
#체크보드
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
ckecpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                            save_best_only= True)

print(iris)

iris = load_iris()

x = iris.data
y = iris.target
print(x.shape)      # (150, 4) #들어가는 게 4!!
print(y.shape)      # (150, )


# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 원핫인코딩
from keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y.shape)          # (150, 3) #나가는 게 3!!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,random_state= 10)


#2. 모델링
model = Sequential()
model.add(Dense(10, input_shape = (4, ), activation = 'relu')) #dnn으로 들어가는 게 4!!
model.add(Dense(20, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(15, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))  #나가는 게 3!!





#3.훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs =50, batch_size= 64,
                validation_split = 0.2, verbose = 2,
                callbacks = [es])

#평가
loss, acc = model.evaluate(x_test, y_test, batch_size = 64)
print('loss: ', loss )
print('acc: ', acc)


import matplotlib.pyplot as plt
plt.figure(figsize = (10, 5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], c= 'red', marker = '^', label = 'loss')
plt.plot(hist.history['val_loss'], c= 'cyan', marker = '^', label = 'val_loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], c= 'red', marker = '^', label = 'acc')
plt.plot(hist.history['val_acc'], c= 'cyan', marker = '^', label = 'val_acc')
plt.title('accuarcy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

