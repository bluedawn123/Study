from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np

#CNN함수 정의
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'valid', 
                    input_shape= (28, 28, 1), activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))

    # model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(filters = hidden_layer_size, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 1, kernel_size = (3, 3), padding = 'valid', activation = 'sigmoid'))
    model.summary()
    return model

#불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

#쉐이프확인
print("x_train.shape : ", x_train.shape)  #(50000, 32, 32, 3)
print("x_test.shape : ", x_test.shape)    #(10000, 32, 32, 3)
print("y_train.shape : ", y_train.shape)  #(50000, 1)
print("y_test.shape : ", y_test.shape)    #(10000, 1)


#원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print("y_train의 shape : ", y_train.shape)   #(50000, 10)  
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")


#데이터전처리 2. x의 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255.                                                                                                                                     
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255.



model = autoencoder(hidden_layer_size = 154)
model.compile(optimize = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(x_train_noised, x_train, epochs = 10)


output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9,ax10),(ax11, ax12, ax13, ax14,ax15)) = \
    plt.subplots(3, 5, figsize=(20, 7))



#원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()