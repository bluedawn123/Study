from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(train_X, train_y)

print(model.score(test_X, test_y))