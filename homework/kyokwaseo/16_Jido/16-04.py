
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


model = Classifier() 


model.fit(train_X, train_y)


model.predict(test_X)

#model.score(test_X, test_y)