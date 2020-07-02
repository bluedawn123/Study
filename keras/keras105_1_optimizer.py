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

optimizer = Adam
optimizer = optimizer(lr = 0.001)
model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])
model.fit(x, y, epochs = 100, verbose = 0)

loss = model.evaluate(x, y)
print('loss 1 : ', loss)           # [0.6082711815834045, 0.6082711815834045]
pred1 = model.predict([3.5])
print('pred 1 : ', pred1)          #[[2.4572725]]


optimizer = RMSprop
optimizer = optimizer(lr = 0.001)
model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])
model.fit(x, y, epochs = 100, verbose = 0)

loss = model.evaluate(x, y)
print('loss 2 : ', loss)
pred1 = model.predict([3.5])      #[0.0004713935195468366, 0.0004713935195468366]
print('pred 2 : ', pred1)         #[[3.473006]]
