from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#데이터
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)

#데이터로 분할
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

#모델
model = LinearSVC()

model.fit(train_X, train_y)


print(model.score(test_X, test_y))