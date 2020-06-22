## 피쳐 엔지니어링

import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

## 데이터
x, y = load_boston(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBRegressor()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("Score : ", score)

thresholds = np.sort(model.feature_importances_)   #fi정렬
print(thresholds)

''' 중요도가 낮은 것들부터 먼저.
0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
오름차순 정렬 
'''
for a in thresholds:               # 컬럼 수만큼 돈다, 빙글빙글
    selection = SelectFromModel(model, threshold = a, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 : ", score)

    print("a = %.3f, n = %d, R2: %.2f%%" %(a, select_x_train.shape[1],
          score * 100.0))


'''
a = 0.001, n = 13, R2: 92.21%
a = 0.004, n = 12, R2: 92.16%
a = 0.012, n = 11, R2: 92.03%
a = 0.012, n = 10, R2: 92.19%
a = 0.014, n = 9, R2: 93.08%
a = 0.015, n = 8, R2: 92.37%
a = 0.018, n = 7, R2: 91.48%
a = 0.030, n = 6, R2: 92.71%
a = 0.042, n = 5, R2: 91.74%
a = 0.052, n = 4, R2: 92.11%
a = 0.069, n = 3, R2: 92.52%
a = 0.301, n = 2, R2: 69.41%
a = 0.428, n = 1, R2: 44.98%
'''