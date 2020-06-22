from sklearn.datasets import load_iris
from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, plot_importance, XGBRegressor
#1. data
iris = load_iris()

x = iris.data
y = iris.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)


print(x.shape) #506, 13
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, 
                                                    random_state = 66)

n_estimators = 83         #나무의 숫자. randomforest는 앙상블. tree모델이 합쳐진것이다. 거기서 업글된게 xgboost
                            #트리구조는 전처리 필요없다. 결측치 제거도 필요없다. 속도가 빠르다
                            #n_estimator은 나무의 숫자. 

learning_rate = 0.01       #학습률?
colsample_bytree = 0.9      #최대치는 1 보통 0.6에서부터 증가시킨다. 
colsample_bylevel = 0.9

n_jobs = -1                 #딥러닝이 아닐 경우 n_job = -1
max_depth = 7      

#추후 CV꼭 쓰고 Feature_importance도 써야한다.

model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate,
                      n_estimators = n_estimators, n_jobs=n_jobs,
                      colsample_bylevel = colsample_bylevel, colsample_bytree =colsample_bytree)

model.fit(x_train, y_train)


score = model.score(x_test, y_test)
print('score : ', score)

print(model.feature_importances_)           #아래의 plot_importance를 보여줄수 있따. 

#print("------------------------------------")
##print(model.best_estimator_)
#print(model.best_params_)
#print("------------------------------------")

#0.9666666666666667
plot_importance(model)
plt.show()


'''
### 3. 모델 훈련
model.fit(x_train, y_train)

### 4. 모델 평가 및 결과 예측
# x_test = mms.inverse_transform(x_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("y의 예측값 : \n", y_pred[:5])
print("모델 정확도 : ", acc)
'''