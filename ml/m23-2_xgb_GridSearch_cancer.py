from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
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
x, y = load_breast_cancer  (return_X_y = True)  #그냥 x,y를 넣어주겠다는 의미
print(x.shape)                  # (569, 30)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, 
                                                    random_state = 66)
'''
n_estimators = 83         #나무의 숫자. randomforest는 앙상블. tree모델이 합쳐진것이다. 거기서 업글된게 xgboost
                            #트리구조는 전처리 필요없다. 결측치 제거도 필요없다. 속도가 빠르다
                            #n_estimator은 나무의 숫자. 

learning_rate = 0.01       #학습률?
colsample_bytree = 0.9      #최대치는 1 보통 0.6에서부터 증가시킨다. 
colsample_bylevel = 0.9

n_jobs = -1                 #딥러닝이 아닐 경우 n_job = -1
max_depth = 7      
'''
parameters = [
               {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.5, 0.01, 0.001], "max_depth" : [4, 5, 6]} ,                                                       
               {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.01, 0.001], "max_depth" : [4, 5, 6],
                "colsample_bytree":[0.6,0.9,1]},
               {"n_estimators" : [90, 110], "learning_rate" : [0.1, 0.001, 0.5], "max_depth" : [4, 5, 6], 
               "colsample_bytree":[0.6,0.9,1],"colsample_bylevel" : [0.6, 0.7, 0.8]}
             ]

n_jobs = -1   

#추후 CV꼭 쓰고 Feature_importance도 써야한다.
model = GridSearchCV(XGBClassifier(), parameters, cv=10, n_jobs=-1)

model.fit(x_train, y_train)
print("------------------------------------")
print(model.best_estimator_)
print(model.best_params_)
print("------------------------------------")

score = model.score(x_test, y_test)
print('score : ', score)                     #score :  0.9736842105263158

#print(model.feature_importances_)           #아래의 plot_importance를 보여줄수 있따. 


#plot_importance(model)
#plt.show()

#여기서 feature_importance와 plt.show는 xgboost에서 먹히는 것이므로 여기선 주석처리.