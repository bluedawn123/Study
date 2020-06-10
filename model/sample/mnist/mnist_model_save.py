import numpy as np

#1. 데이터
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import History, ModelCheckpoint, EarlyStopping


modelpath = './model/sample/minst/check-{epoch:02d} - {val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto', save_weights_only = False, verbose = 1)


es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')





mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                    # (60000, 28, 28)
print(x_test.shape)                    # (10000, )
print(y_train.shape)                    # (60000, )
print(y_test.shape)                    # (10000, )

print("x_train 모양 : ", x_train.shape)                 
print("x_test 모양 : ", x_test.shape)                  
print("y_train 모양 : ", y_train.shape)                   
print("y_test 모양 : ", y_test.shape) 


# x_data 전처리 : Dense형 모델 사용을 위한 '2차원' reshape
x_train = x_train.reshape(60000, 28*28 ).astype('float32') / 255.
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.
print(x_train.shape)                  # (60000, 784)
print(x_test.shape)                  # (10000, 784)


# y_data 전처리 : one_hot_encoding (다중 분류)
from keras.utils.np_utils import to_categorical
#from keras.utils import np_utils 는 필요가 없는가?
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("y_train의 shape : ", y_train.shape)
print("y_test의 shape : ", y_test.shape)




#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (28*28, )))   #input_shape = (28 * 28, ) 로 대신해도 괜찮다. 
model.add(Dense(30, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(18, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


#model.save('./model/sample/mnist/model_test01.h5')  #이러면 모델까지만 저장.



#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'] )
model.fit(x_train, y_train, epochs = 5, batch_size = 32, shuffle = True,
                            validation_split =0.2, callbacks = [es, cp])


#model.save_weights('./model/sample/mnist/test_weight1.h5')


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size= 32)
print('loss: ', loss)
print('acc: ', acc)


