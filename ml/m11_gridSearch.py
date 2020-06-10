import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.svm import SVC
#1데이터
iris = pd.read_csv('./data/csv/iris.csv')

x = iris.iloc[:, 0:4]         #판다스니깐~~ loc, iloc는 행,열(헤더, 인덱스)가 있어야 한다. 
y = iris.iloc[:, 4]
print("x.shape : ", x.shape)  #150, 4
print("y.shape : ", y.shape)  #150, 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)

parameters = [{"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},
              {"C" : [1, 10, 100, 1000], "kernel" : ["rbf"], "gamma" : [0.001, 0.0001]},
              {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma" : [0.001, 0.0001]}
]

kfold = KFold(n_splits = 5, shuffle = True)
model = GridSearchCV(SVC(), parameters, cv = kfold)  #svc라는 모델의 파라미터 cv=cross validation
                                                     #train을 20퍼를 검증때 쓰겠다. 

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("최종 정답률 : ", accuracy_score(y_test, y_pred))

print("최적의 매개변수 : ", model.best_estimator_)
 

#c = 1, gamma = none, kernel = linear 일때 최상이다. 