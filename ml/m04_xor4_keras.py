from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor,
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Flatten, Dropout
#from keras.callbacks import EarlyStopping

import numpy as np

#1. 데이터
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])


print("x_data의 형태 : ", x_data.shape)  #  (4, 2)
print("y_data의 형태 : ", y_data.shape)  #  (4, )

#2.모델
#model = LinearSVC()
#model = SVC()
#model = KNeighborsClassifier(n_neighbors=1)
#model = KNeighborsRegressor
#model = RandomForestClassifier
#model = RandomForestRegressor


model = Sequential() 
model.add(Dense(5, input_dim=2, activation='relu'))       #(들어가는 게 2, 나가는 게 1)
model.add(Dense(15, activation = 'relu'))
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
model.add(Dense(1, activation = 'sigmoid'))  #y값에서 나가는 게 3


#3.실행
#early_stopping = EarlyStopping(monitor='loss', patience=15, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs =500, verbose=1, batch_size=1)#callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x_data, y_data, batch_size=1) 

x_test = x_data

print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_test)
print("y의 예측값 : ", y_pred)  #별 의미가 없다. 

'''

early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') #monitor은 출력의미
model.fit([x1_train, x2_train],
          y1_train, 
          epochs=200, batch_size=1, 
          validation_split=(0.25), verbose=1,
          callbacks=[early_stopping])  #기본이 리스트
          
'''