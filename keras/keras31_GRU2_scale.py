#lstm3_scale을 gru2_scale로 변경


from numpy import array
from keras.models import Sequential
from keras.layers import Dense, GRU

# 1. 데이터 구성
x = array([[1, 2, 3,], [2, 3, 4,], [3, 4, 5], [4, 5, 6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40],[30,40,50],[40,50,60]])  #(13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])                                    #(13, )

#이 경우 weight 와 bias는 ??


x_predict = array([50, 60, 70])  


#z = array([[4,5,6,7]])
#q = array([[4], [5]])
#(뭐, 뭐)는 중괄호 기준 + 작은것부터 역으로 본다!!

print("x.shape : ", x.shape)        # res : (4, 3)
print("y.shape : ", y.shape)        # res : (4, ) 그냥 스칼라 4개.
#print("z.shape : ", z.shape)           
#print("q.shape : ", q.shape)  

#☆☆☆☆☆☆☆ 여기서 (4, 3)을 (4, 3, 1)로 바꿔줘야 한다. 

#????????????뭔소리임?!?!?
#x = x.reshape(4, 3, 1)                       #아래거랑 같다는데 무슨소리?!?!?!? 왜 이렇게 변경하는 거임?!?!?!?
x = x.reshape(x.shape[0], x.shape[1], 1)   #전체를 곱하면 이상이 없다 ??!! 무슨소리 ?!?!?  ☆두개가 같다. ☆

''' 질문 2                         행            열        몇개씩 자르는지
#x의 shape 의 의미 => (batch_size,      timesteps,        feature)
input_shape = (timesteps, feature)
input_length = timesteps, input_dim = feature

즉 배치사이즈는 행의 크기대로 자르겠다. 피쳐는 LSTM의 안쪽의 갯수를 몇개씩 자르겠다는 의미!



'''
print(x.shape)          #(13, 3, 1)로 맞춰줌!!!            

#☆☆☆☆☆☆☆☆☆☆질문 1. (13, 3, 1)로 맞춰줬으나 이것을 어디에 넣는거임? 혹은 어디서 어떻게 훈련을 시키는 거임?!?!?!☆☆☆☆☆☆☆

#2. 모델구성  ()
model = Sequential()
#model.add(LSTM(10, activation = 'relu', input_shape=(3,1)))  #인풋쉐이프가 3,1의 의미?? >>  #4,3,1에서 4무시. (3,1)을 모델의 기준으로 잡겠다. 


model.add(GRU(20, input_length=3, input_dim=1))   #☆☆☆☆☆☆☆질문1. input_dim 의 의미..? 입력이 한개.(왜냐하면, [1],[2],[3] 이런식으로 만들어줬으니)
                                                                                          # 
                                                   #☆☆☆☆☆☆☆질문2. 바로 윗줄과의 차이..?
                                                   ##☆☆☆☆☆☆☆질문3. input_length = 3은 3개의 열이라고 생각할 수 있다. 위의 모양이 (4, 3, 1)이므로
model.add(Dense(8))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(1115))
model.add(Dense(35))
model.add(Dense(23))
model.add(Dense(17))
model.add(Dense(12))
model.add(Dense(1))   #☆☆☆☆☆☆☆질문 2. 1인 이유?!?!?!?!?☆☆☆☆☆☆☆

model.summary()


#3. 실행
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x,y, epochs = 200)

x_predict = array([50, 60, 70])     #그냥 (3, ). 그냥 스칼라3개, 벡터 1개
                            #    여기서 5,6,7을 넣는 의미?!?!?!?!?!?    >>>>>> [5,6,7] 을 넣어서 다음 y값을 예측하겠다..!!

x_predict = x_predict.reshape(1,3,1)  #☆☆☆☆☆☆☆질문3. 왜 여기서 (1, 3, 1)로 변경하는 거임?!?!?!?!?☆☆☆☆☆☆☆




#4. 예측
print(x_predict)  #(1,3,1) 완성. 그러므로 이걸 모델에 집어 넣을 수 있다. 
 
#☆☆☆☆☆☆☆질문4. 이것을 어디에 넣고 훈련(혹은 평가예측)해서 y의 예측값을 추출하는 거임?!?!?☆☆☆☆☆☆☆

y_predict = model.predict(x_predict)
print(y_predict)     #최종적으로 y의 예측값인 yhat을 출력한다.!!


###우리는 지금, LSTM은 수정을 못한다. 시킬 수 있는 부분은?

