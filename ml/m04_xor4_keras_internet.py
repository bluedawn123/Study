'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

training_data = np.array([[0,0],[0,1],[1,0],[1,1]])

target_data = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['binary_accuracy'])


model.fit(training_data, target_data, nb_epoch=500, verbose=1)
'''

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, nb_epoch=500, verbose=1)



'''
while True:
    inp = list(map(int,input().split()))
    qwe = np.array(inp)
    print("입력된 값 : {}".format(qwe),end="\n\n")
    qwe = qwe.reshape(1,2)
    print("reshape : {}".format(qwe),end="\n\n")
    print("결과값 : {}" .format(model.predict(qwe)[0][0].round()))
    print("")
    '''