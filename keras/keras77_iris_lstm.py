import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Flatten, Dropout, LSTM, Input
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 다중 분류
from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target
print(x.shape)      # (150, 4)
print(y.shape)      # (150, )


# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x).reshape(150, 4, 1)                #행무시.. 그러므로 4, 1이 들어간다. 

# 원핫인코딩
from keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y.shape)          # (150, 3)  #다중분류 후 와이 형태 확인

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,random_state= 10)


#2. model
model = Sequential()
model.add(LSTM(10, input_shape = (4,  1), activation = 'relu'))  #LSTM이므로.!! 
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(15, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(34, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(45, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation = 'softmax'))  #나가는 게 3


# callbacks 
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
# earlystopping
es = EarlyStopping(monitor = 'val_loss', patience = 25, verbose =1)
# Tensorboard
ts_board = TensorBoard(log_dir = 'graph', histogram_freq= 0,
                      write_graph = True, write_images=True)
# Checkpoint
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                            save_best_only= True)


#3. compile, fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs =30, batch_size= 32,
                validation_split = 0.2, verbose = 2,
                callbacks = [es, ts_board, checkpoint])

# evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size = 64)
print('loss: ', loss )
print('acc: ', acc)

# graph
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

plt.show()

