from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten #4차원(장수,가로,세로,명암), 



model = Sequential()
model.add(Conv2D(10, (2,2),  input_shape=(10, 10, 1)))    # (2,2) = 픽셀을 2 by 2 씩 잘른다.    #(가로,세로,명암 1=흑백, 3=칼라)(행, 열 ,채널수) # batch_size, height, width, channels
                                                                     #(9, 9, 10)           
model.add(Conv2D(7, (3,3)))                                          #(7, 7, 7)
model.add(Conv2D(5, (2,2), padding='same'))                          #(7, 7, 5)
model.add(Conv2D(5, (2,2)))                                          #(6, 6, 5)               
#model.add(Conv2D(5, (2,2), strides = 2))                   #☆☆☆왜 3,3,5인지 알기!          #(3, 3, 5)  ☆☆☆
#model.add(Conv2D(5, (2,2), strides = 2, padding='same'))                             #(3, 3, 5) ☆☆☆3, 3, 5 인 이유는 스트라이드가 우선순위이기 때문이다.   

model.add(MaxPooling2D(pool_size = 2))                      
model.add(Flatten())   #데이터를 편다. 
model.add(Dense(1))   # 이 이유?                                      #(n, 3, 3, 1)


model.summary()


