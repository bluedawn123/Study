import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)

#데이터로 분할
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

#모델
model = LinearSVC()

model.fit(train_X, train_y)



plt.scatter(X[:, 0], X[:, 1], c=y, marker=".", cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)


Xi = np.linspace(-10, 10)
Y = -model.coef_[0][0] / model.coef_[0][1] * \
    Xi - model.intercept_ / model.coef_[0][1]

# 그래프
plt.plot(Xi, Y)

# 그래프 스케일을 조정
plt.xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
plt.ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
plt.axes().set_aspect("equal", "datalim")

# 그래프제목설정
plt.title("classification data using LinearSVC")

# x축, y축 이름
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()