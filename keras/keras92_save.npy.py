from sklearn.datasets import load_iris
import numpy as np


iris = load_iris()

print(type(iris))  #<class 'sklearn.utils.Bunch'> 이 자체로만 사용할 수 없다. 

x_data = iris.data
y_data = iris.target


print("x의 데이터의 타입 : ", type(x_data))  #<class 'numpy.ndarray'>
print("y의 데이터의 타입 : ", type(y_data))  #<class 'numpy.ndarray'>

#위의 numpy를 저장할 수 있따.

np.save('./data/iris_x.npy', arr= x_data)
np.save('./data/iris_y.npy', arr= y_data)  #train test 분류전

#data파일에 2개가 생기면 넘파이로 저장이 잘 됐다는 것이다. 그럼 불러와보자

x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load))
print(type(y_data_load))

print("x_data_load.shape의 모양 : ", x_data_load.shape)  # (150, 4)
print("x_data_load.shape의 모양 : ", y_data_load.shape)  # (150, )













