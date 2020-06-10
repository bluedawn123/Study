#kears  53 복붙!

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import History

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0].shape)
print("y_train: ", y_train[0])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_train[0].shape)
#plt.imshow(x_train[0], 'gray')
#plt.imshow(x_train[0])
#plt.show()

#  데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 2.정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255 # 255 검정 0 백색
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255


from keras.models import load_model
model = load_model('./model/model_test01.h5')





#85번에 이어서 3layer 추가해서 연결.






model.summary()



#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) #x,y를 평가하여 loss와 acc에 반환하겠다.






print('loss, : ', loss)
print('loss, : ', acc)




'''

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)


plt.plot(hist.history['loss'], marker = '.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['loss', 'val_loss'])
plt.legend(loc = 'upper right')
plt.show()

plt.subplot(2, 1, 2)

#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()


'''


'''
x_pred = (x_test)
y_predict = model.predict(x_pred)
print('y_pred :', y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

'''



