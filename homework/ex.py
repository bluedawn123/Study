from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Flatten, Dropout, LSTM, Input
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler 

from sklearn.datasets import load_iris





(x_train, y_train), (x_test, y_test) = iris.load_data()

print("x_train[0] : ", x_train[0])
print("y_train[0] : ", y_train[0])

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ", x_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

plt.imshow(x_train[0])
plt.show()



