# 그냥 예시..
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 데이터 생성
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)


train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

#모델 생성
model = LogisticRegression()

#학습
model.fit(train_X, train_y)

#test_X로 y예측
pred_y = model.predict(test_X)

###################################################################################
plt.scatter(X[:, 0], X[:, 1], c=y, marker=".",
            cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)


Xi = np.linspace(-10, 10)
Y = -model.coef_[0][0] / model.coef_[0][1] * \
    Xi - model.intercept_ / model.coef_[0][1]
plt.plot(Xi, Y)



plt.xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
plt.ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
plt.axes().set_aspect("equal", "datalim")
#######################################################################################
# 그래프에 제목을 설정
plt.title("classification data using LogisticRegression")

# x축, y축에 각각 이름 설정
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()