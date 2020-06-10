from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten #4차원(장수,가로,세로,명암), 



model = Sequential()
model.add(Conv2D(10, (2,2),  input_shape=(15, 15, 1)))    
                                                                      
model.add(Conv2D(7, (3,3)))                                          
model.add(Conv2D(5, (2,2), padding='same'))                        
model.add(Conv2D(5, (2,2)))                                                    
model.add(Conv2D(5, (2,2), strides = 2))                   
model.add(Conv2D(5, (2,2), strides = 2, padding='same'))                      

model.add(MaxPooling2D(pool_size = 2))                      
model.add(Flatten())   
model.add(Dense(1))   
model.summary()


