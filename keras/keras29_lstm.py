from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 구성
x = array([[1, 2, 3,], [2, 3, 4,], [3, 4, 5], [4, 5, 6]])  #(4,3)
y = array([4, 5, 6, 7])                                    #(4, )

print("x.shape : ", x.shape)        # res : (4, 3)
print("y.shape : ", y.shape)        # res : (4, ) 그냥 스칼라 4개.

#☆☆☆☆☆☆☆ 여기서 (4, 3)을 (4, 3, 1)로 바꿔줘야 한다. 


#x = x.reshape(4, 3, 1)                       
x = x.reshape(x.shape[0], x.shape[1], 1)   #☆두개가 같다. ☆


print(x.shape)          #(4, 3, 1)로 맞춰줌!!!            

#☆☆☆☆☆☆☆☆☆☆질문 1. (4, 3, 1)로 맞춰줬으나 이것을 어디에 넣는거임? 혹은 어디서 어떻게 훈련을 시키는 거임?!?!?!☆☆☆☆☆☆☆

#2. 모델구성  ()
model = Sequential()

model.add(LSTM(10, activation = 'relu', input_shape=(3,1)))  #인풋쉐이프가 3,1의 의미?? >>  #4,3,1에서 4무시. (3,1)을 모델의 기준으로 잡겠다. 
model.add(Dense(5))
model.add(Dense(1))   #☆☆☆☆☆☆☆질문 2. 1인 이유?!?!?!?!?☆☆☆☆☆☆☆

model.summary()


#3. 실행
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x,y, epochs = 180, )

x_input = array([5, 6, 7])     #그냥 (3, ). 그냥 스칼라3개, 벡터 1개
                            #    여기서 5,6,7을 넣는 의미?!?!?!?!?!?    >>>>>> [5,6,7] 을 넣어서 다음 y값을 예측하겠다..!!


x_input = x_input.reshape(1,3,1)  #☆☆☆☆☆☆☆질문3. 왜 여기서 (1, 3, 1)로 변경하는 거임?!?!?!?!?☆☆☆☆☆☆☆



print(x_input)  #(1,3,1) 완성. 그러므로 이걸 모델에 집어 넣을 수 있다. 
 
#☆☆☆☆☆☆☆질문4. 이것을 어디에 넣고 훈련(혹은 평가예측)해서 y의 예측값을 추출하는 거임?!?!?☆☆☆☆☆☆☆

yhat = model.predict(x_input)
print(yhat)     #최종적으로 y의 예측값인 yhat을 출력한다.!!


###우리는 지금, LSTM은 수정을 못한다. 시킬 수 있는 부분은?