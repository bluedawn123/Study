from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#mglearn.plots.plot_linear_regression_wave()  #w[0]: 0.393906  b: -0.031804
#plt.show()


X, y = mglearn.datasets.make_wave(n_samples=60)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_:, 즉 가중치 : {}".format(lr.coef_))
print("lr.intercept_:,즉 절편 : {}".format(lr.intercept_))

print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))