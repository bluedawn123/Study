import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

x= np.array([1,2,3,4])
y=np.array([1,2,3,4]) 

#모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

optimize = [Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam]

for optimizer in optimize:
    optimizer = optimizer(lr = 0.001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])

    model.fit(x, y, epochs = 100, verbose = 0)

    loss = model.evaluate(x, y)
    print('loss : ', loss)

    pred1 = model.predict([3.5])
    print(pred1)

'''
#loss :  [0.027571965008974075, 0.027571965008974075]
[[3.4601505]]
4/4 [==============================] - 0s 3ms/step
loss :  [0.0007298654527403414, 0.0007298654527403414]
[[3.531968]]
4/4 [==============================] - 0s 3ms/step
loss :  [1.1328309483360499e-05, 1.1328309483360499e-05]
[[3.498487]]
4/4 [==============================] - 0s 3ms/step
loss :  [9.006902473629452e-06, 9.006902473629452e-06]
[[3.4986925]]
4/4 [==============================] - 0s 3ms/step
loss :  [1.2067381760516582e-08, 1.2067381760516582e-08]
[[3.5000546]]
4/4 [==============================] - 0s 4ms/step
loss :  [5.182867971598171e-05, 5.182867971598171e-05]
[[3.5089812]]
'''