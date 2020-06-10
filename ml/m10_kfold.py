import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')


#1데이터
iris = pd.read_csv('./data/csv/iris.csv')

x = iris.iloc[:, 0:4]         #판다스니깐~~ loc, iloc는 행,열(헤더, 인덱스)가 있어야 한다. 
y = iris.iloc[:, 4]
print(x)
print(y)


warnings.filterwarnings('ignore')

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)


kfold = kfold(n_split=5, shuffle = True)  #kfold를 5등분으로 하겠고 섞겠다. 즉, 개당 20퍼씩 쓴다. 

warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter = 'classifier')  #의미..?!?!?

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    scores = cross_val_score(model, x, y, cv=kfold)  #이 한줄로, 5개로 자르고 계속 점수를 내주겠다. 


    print(name, "의 정답률 : ")
    print(scores)


import sklearn
print(sklearn.__version__)



