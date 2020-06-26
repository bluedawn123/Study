from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)


train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)


model = LogisticRegression(random_state=42)


model.fit(train_X, train_y)


pred_y = model.predict(test_X)
print(pred_y)
