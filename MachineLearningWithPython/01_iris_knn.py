from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

from sklearn.datasets import load_iris
iris = load_iris()

print(iris)
#데이터 이해하기
print(" 키 값들 : ", iris.keys())

#print("키 값인 data 보기 : ", iris['data'])      #[ , , , ,] 씩 나온다.
#print("키 값인 target 보기 : ", iris['target'])   #0,1,2 씩 나온다.

print("키 값인 target_names 보기 : ", iris['target_names']) 
 #['setosa' 'versicolor' 'virginica']로 분류됨을 알 수 있다. 


#즉 우리는 data에 들어가는 4가지 값으로 그것이 3가지 중 어느 곳에 들어가는 지 안다.

x = iris.data
y = iris.target
print(x.shape)      #(150, 4)  들어가는 게 4
print(y.shape)      #(150,)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x,y, random_state=0)

print("X_train.shape : ", X_train.shape)
print("y_train.shape : ", y_train.shape)
print("X_test.shape : ", X_test.shape)
print("y_test.shape : ", y_test.shape)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

'''
#예를들어 X_new를 만들어 넣어보다.
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)
print("예측:", prediction)
print("예측한 타깃의 이름:", 
       iris['target_names'][prediction])
'''

y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)

score = knn.score(X_test, y_test)
print("score : ", score)
print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))

X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test, y_test)))