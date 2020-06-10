from sklearn.datasets import load_iris
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Parallel, Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#1. data
iris = load_iris()

x = iris.data
y = iris.target

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, train_size =0.8, shuffle=True)

print(x.shape)
print(y.shape)

#모델
#pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])   #SVC와 MinMAx를 쓰겠다. 
pipe = make_pipeline(MinMaxScaler(), SVC())                      #make_pipeline을 하면 전처리와 모델만 써주면 된다. 

pipe.fit(x_train, y_train)

print("acc : ", pipe.score(x_test, y_test))























