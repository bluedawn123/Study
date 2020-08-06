from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.datasets import mnist

def autoencoder(hidden_layere_size):
    model = Sequential()
    model.add(Conv2D(filters= 32, kernal_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
    model.add(Conv2D(7,(3,3)))    #strides : 높이와 너비를 따라 컨벌루션의 보폭을 지정하는 정수 또는 튜플 / 2 개의 정수 목록입니다. 모든 공간 치수에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다. 모든 보폭 값! = 1을 지정하면 모든 dilation_rate값! = 1 을 지정할 수 없습니다 .
    model.add(Conv2D(5,(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(7,(2,2)))
    model.add(Conv2D(9,(2,2)))
    model.add(Conv2D(5,(2,2),strides=2, padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten()) # 2차원으로 변경  >>>>>>>>>>> 즉 DNN을 구성할 수 있따. 
    model.add(Dense(10,activation='softmax'))
    model.summary()
    return model


train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print(x_train.shape[0]) #60000
print(x_train.shape[1]) #28
print(x_train.shape[2]) #28


# model = autoencoder(hidden_layere_size=32)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

x_train = x_train/255.
x_test = x_test/255.

model = autoencoder(hidden_layere_size=154)


# model.compile(optimizer='adam', loss='mse', metrics=['acc']) # loss = 0.0119
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])    #0.08

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다. 
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0 : 
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0 : 
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

