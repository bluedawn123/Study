import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()
pca = PCA(n_components = 3)


### 1. 데이터
x, y = load_iris(return_X_y = True)
print(x.shape)                  # (150, 4)
print(y.shape)                  # (150,)

## 1-1. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)            # (120, 4)
print(x_test.shape)             # (30, 4)
print(y_train.shape)            # (120,)
print(y_test.shape)             # (30,)

## 1-2. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])               # [0. 1. 0.]
print(y_test[0])                # [0. 1. 0.]

## 1-3. PCA
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)            # (120, 3)
print(x_test.shape)             # (30, 3)

## 1-4. Scaler
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)
print(x_train[0])               # [0.78879436 0.55721313 0.68381312]
print(x_test[0])                # [0.46608106 0.25743542 0.32323374]


### 2. 모델링
rfc = RandomForestClassifier(n_estimators = 100, max_depth = 7)

model = rfc


### 3. 훈련
model.fit(x_train, y_train)


### 4. 모델 평가
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("y의 예측값 : ", y_pred[:5])
print("모델 정확도 : ", acc)


