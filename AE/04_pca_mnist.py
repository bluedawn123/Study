from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Datasets 불러오기
from tensorflow.keras.datasets import mnist  

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])


print('y_train : ' , y_train[0])


print(x_train.shape)                   #(60000, 28, 28)
print(x_test.shape)                    #(10000, 28, 28)
print(y_train.shape)                   #(60000,) 스칼라, 1 dim(vector)
print(y_test.shape)                    #(10000,)

print(x_train[0].shape) #(28,28) 짜리 

# 1. y-------------------------------------------------------------------
# Data 전처리 / 1. OneHotEncoding 큰 값만 불러온다 y
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10) 6만장 10개로 증폭/ 아웃풋 dimension 10

# 0과 255 사이를 0과 1 사이로 바꿔줘

x_train = x_train.reshape(60000, 784).astype('float32')/255 ##??????
x_test  = x_test.reshape (10000, 784).astype('float32')/255 ##??????

print('x_train.shape: ', x_train.shape)  #(60000, 784)
print('x_test.shape : ' , x_test.shape)  #(10000, 784)

X = np.append(x_train, x_test, axis = 0)

print(X.shape)  #(70000, 784)

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
aaa = np.argmax(cumsum >= 0.99)+1 
print(aaa)  #331  즉, 99퍼 이상의 압축률을 갖은 pca쓰려면 331번째부터. 
















