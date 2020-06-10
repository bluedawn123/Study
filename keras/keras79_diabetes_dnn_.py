from sklearn.datasets import load_diabetes
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 1. 데이터
a_data = load_diabetes()
#print(f"1data : {a_data}")
print("1. data : ",a_data)

#print("2a_data.keys() : ",a_data.keys())  #☆☆☆☆ a_data의 키 값을 볼 수 있다!!
print("데이터의 키 값 : ", a_data.keys())  #☆☆☆☆ a_data의 키 값을 볼 수 있다!!
#위의 것을 출력하면,  dict_keys(['data', 'target', 'DESCR', 'feature_names', 
#                               'data_filename', 'target_filename'])
#이 나온다. 여기서 y값의 유추 할 수 있다. 

print(f"data.type : {type(a_data)}")

print("------------------------------------------------------------")

x_data = a_data.data 
# print(f"x_train : {x_train}")
print(f"x_data.shape : {x_data.shape}") # (442, 10)
print("------------------------------------------------------------")
y_data =a_data.target                                     #target을 알 수 있는 방법? 위 서술
# print(f"y_train : {y_train}")
print(f"y_data.shape : {y_data.shape}") # (442,)



print("------------------------------------------------------------")

feature_names = a_data.feature_names                     #feature_names를 알 수 있는 방법? 위
#설명 DESCR  = a_data.DESCR                     #feature_names를 알 수 있는 방법? 위
print(f"feature_names : {feature_names}") # 10개의 칼럼
#print(f"DESCR : {DESCR}") # 10개의 칼럼


#정규화
std = StandardScaler()
std.fit(x_data) # (,)
x_data = std.transform(x_data)

print(" ")
print("------------------------------------------------------------")
# pca = PCA(n_components=9)
# pca.fit(x_data)

# x_data = pca.fit_transform(x_data)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print('선택할 차원 수 :', d)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split( 
    x_data,y_data,random_state = 66, shuffle=False,
    train_size=0.8
    )

print(f"x_train.shape : {x_train.shape}") # x_train.shape : (442,10)   


# 2. 모델
model = Sequential()
model.add(Dense(64,input_shape=(10,)))  
model.add(Dropout(0.6))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.6))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.6))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()




# 3. 컴파일(훈련준비),실행(훈련)
model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

hist = model.fit(x_train,y_train,epochs=25,batch_size=3,callbacks=[],verbose=2,validation_split=0.03)

# 4. 평가, 예측





# R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2_y_predict = r2_score(y_test,y_predict)

plt.figure(figsize=(10,6)) # -> 도화지의 크기? 출력되는 창의 크기인가 그래프의 크기인가 

plt.subplot(2,1,1) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras79 loss plot')
plt.plot(hist.history['loss'],marker='.', c='red',label = 'loss') 
plt.plot(hist.history['val_loss'],marker='.', c='blue',label = 'val_loss')

plt. grid()

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2) # 2행1열의 첫번쨰 그림을 그린다.
plt.title('keras79 mse plot')

plt.plot(hist.history['val_mse'])
plt.plot(hist.history['mse'])

plt. grid()

plt.ylabel('mse')
plt.xlabel('epoch')

plt.legend(['train mse','val mse'])

plt.show()
loss,mse = model.evaluate(x_test,y_test,batch_size=3)

print(f"mse : {mse}") 
print(f"loss : {loss}")
print("r2 : ",r2_y_predict)

