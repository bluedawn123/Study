#선형 SVM으로 데이터의 분류를 학습하고, test_X와 test_y를 사용하여 모델의 정확도를 출력하시오.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 데이터 
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 모델 구축
model = LinearSVC()

# train_X와 train_y
model.fit(train_X, train_y)

# test_X와 test_y모델의 정확도
print(model.score(test_X, test_y))


plt.scatter(X[:, 0], X[:, 1], c=y, marker=".",
            cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)


Xi = np.linspace(-10, 10)
Y = -model.coef_[0][0] / model.coef_[0][1] * Xi - model.intercept_ / model.coef_[0][1]
plt.plot(Xi, Y)


plt.xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
plt.ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
plt.axes().set_aspect("equal", "datalim")


plt.title("classification data using LinearSVC")


plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()
