import numpy as np
from keras.models import Sequential #Sequential:순차적 모델을 만든다.
from keras.layers import Dense #Dense:1차함수

x = np.array(range(1,11))
# y = np.array([1,2,3,4,5,1,2,3,4,5])
y = np.array([1,2,3,4,5,1,2,3,4,5])



print("x.shape : ", x.shape)                #(10, )          input_dim = 1
print("y.shape : ", y.shape)                #(10, )          output_dim = 1


from keras.utils import np_utils
y = np_utils.to_categorical(y)


print(x)
print(y)
print("y.shape : ", y.shape)


#2. 모델구성

model = Sequential() #모델 구성 순차적
model.add(Dense(5, input_dim =1, activation='relu')) 
model.add(Dense(7, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20))
model.add(Dense(13, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.summary()


#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #손실을 줄이기위해 mse 사용 adam=최적화 metrics=훈련상황을 모니터링
model.fit(x,y,epochs=50,batch_size = 1) #x와 y룰 훈련 /epochs:몇번훈련시킬것인가 /batch size:몇개씩 잘를것인가

#4. 평가 예측
loss, acc= model.evaluate(x,y,batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)

y_pred = np.argmax(y_pred, axis=1)+1
print('y_pred : ', y_pred)
