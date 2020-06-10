#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(311,411), range(100)])     
y1 = np.array([range(711,811), range(711,811), range(100)])    

x2 = np.array([range(101,201), range(411,511), range(100, 200)])
y2 = np.array([range(501,601), range(711,811), range(100)])
print(x1.shape) #  열우선, 행무시
print(y2.shape)



x1 = np.transpose(x1)  #100행 3열.
y1 = np.transpose(y1)
y2 = np.transpose(y2)
x2 = np.transpose(x2)

print(x1.shape)
print(y1.shape)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=False, train_size=0.8)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=False, train_size=0.8)


#2. 모델구성. 모델 2개를 만들거니깐 shape가 2개가 필요하다. sequential로 불가능. 
from keras.models import Sequential, Model #함수형모델을 쓰겠다.  
from keras.layers import Dense, Input #인풋명시를 해야한다. 

#model = Sequential()
#model.add(Dense(5, input_dim = 3))
#model.add(Dense(4))
#model.add(Dense(1))
#함수형 모델로 변경
#인풋, 아웃풋이 뭔지 명시해야한다.

#모델1
input1 = Input(shape=(3, )) #인풋 3
dense1_1 = Dense(5, activation='relu')(input1) #꼬리에 모델명을 붙여준다. #1번째 히든레이어 완성. #input을 명시해야한다. 여기서는 input1
dense1_2 = Dense(4, activation='relu')(dense1_1)
dense1_3 = Dense(7, activation='relu')(dense1_2)


#모델2
input2 = Input(shape=(3, )) #함수형 모델은 순차모델과 다르게 함수 명시를해줘야한다. #함수형 모델은 기본적으로 input shape 로 한다. , 레이어는 이름을 지정해줘야하므로 input1으로 지정
dense2_1 = Dense(5, activation='relu')(input2) #꼬리에 모델명을 붙여준다. #1번째 히든레이어 완성. #input을 명시해야한다. 여기서는 input1
dense2_2 = Dense(6, activation='relu')(dense2_1)
dense2_3 = Dense(3, activation='relu')(dense2_2)


from keras.layers.merge import concatenate #단순병합
merge1 = concatenate([dense1_3, dense2_3]) #2개 이상은list [] 사용...

middle1 = Dense(30)(merge1)  #노드 30개 생성
middle2 = Dense(5)(middle1)
middle3 = Dense(4)(middle2) 

#아웃풋 구성해야한다. 이제.
#3개짜리 모델2개이다. 아웃풋도 2개이므로, 이것을 만들어줘야한다. 
#output 모델구성

output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2) #3으로 나간다. 

output1 = Dense(30)(middle3)
output2_2 = Dense(7)(output1)
output2_3 = Dense(3)(output2_2) #3으로 나간다. 



model = Model(inputs = [input1, input2], outputs= [output1_3, output2_3]) #함수형 모델 명시(input1부터시작해서 output1까지 함수형 모델임을 명시) #list로 구성. 

model.summary() #함수형모델


