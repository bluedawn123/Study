import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist  #keras.dataset => 예제파일 소환

(x_train, y_train), (x_test, y_test) = mnist.load_data()  #mnist에 분류가 되어있다. 소환

print(x_train[255])
print("y_train :", y_train[0])


#14 ~ 17번 질문

print("x_train의 모양 ", x_train.shape)      # (60000, 28, 28)
print("x_test의 모양 ", x_test.shape)        # (10000, 28, 28) 
print("y_train의 모양 ", y_train.shape)      # 1디멘션 (60000,)
print("y_test의 모양 ", y_test.shape)        #(10000,)


print(x_train[1].shape)  #(28,28) 

plt.imshow(x_train[0], 'gray')   #imshow는 그것에 대한 이미지를 볼 수 있다. 
#plt.imshow(x_train[0])
plt.show()








