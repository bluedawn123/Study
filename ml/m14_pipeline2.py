# RandomizedSearchCV + Pipeline
from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Parallel, Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV


# 1. 데이터
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state=43)

# 그리드 / 랜덤 서치에서 사용할 매개 변수
parameters = [
    {"svc__C" :[1, 10, 100, 1000], "svc__kernel" :['linear']},
    {"svc__C" :[1, 10, 100, 1000], "svc__kernel" :['rbf'], 'svc__gamma':[0.001, 0.0001]},
    {"svc__C" :[1, 10, 100, 1000], "svc__kernel" :['sigmoid'], 'svc__gamma':[0.001, 0.0001]}
]
#여기서 svm__C 는 아래의 파이프라인에 명시해 준 것과 모델명이 같아야한다. 'svm'과 같이. 

'''
parameters = [
    {"C" :[1, 10, 100, 1000], "kernel" :['linear']},
    {"C" :[1, 10, 100, 1000], "kernel" :['rbf'], 'gamma':[0.001, 0.0001]},
    {"C" :[1, 10, 100, 1000], "kernel" :['sigmoid'], 'gamma':[0.001, 0.0001]}
]

이 경우에는 에러가 뜬다. 모델명에 대한 변수가 지정하지 않았기 때문. 하지만 위처럼 svm 등등이 있으면 가능하다. 
'''


# 2. 모델
# model = SVC()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])    #여기서 svm과 parameters의 svc, svm등을 맞춰줘야 한다. 
pipe = make_pipeline(MinMaxScaler(), SVC())                       #make_pipeline을 하면 전처리와 모델만 써주면 된다. 
model = RandomizedSearchCV(pipe, parameters, cv=5)






#3. 훈련
model.fit(x_train, y_train)





#4. 평가, 예측
acc = model.score(x_test, y_test)

print("최적의 매개변수 = ", model.best_estimator_)
print("acc : ", acc)

# pipe.fit(x_train, y_train)

# print("acc : ", pipe.score(x_test, y_test))

import sklearn as sk
print("sklearn : ", sk.__version__)

