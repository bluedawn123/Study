from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("선형 훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))  #0.95
print("선형 테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))  #0.61

ridge = Ridge().fit(X_train, y_train)
print("릿지 훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))  #0.89
print("릿지 테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))  #0.75

