# xgboost evaluate

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_breast_cancer

## 데이터
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBRegressor(n_estimators = 300,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = ['error', 'auc'],
          eval_set = [(x_train, y_train),
                      (x_test, y_test)])
        #   early_stopping_rounds = 100)
# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

results = model.evals_result()
print("eval's result : ", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
#print("acc Score : %.2f%%" %(acc * 100))
print("acc : ", acc)

## 그래프, 시각화
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label = 'Train')
ax.plot(x_axis, results['validation_1']['error'], label = 'Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost error')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label = 'Train')
ax.plot(x_axis, results['validation_1']['auc'], label = 'Test')
ax.legend()
plt.ylabel('auc')
plt.title('XGBoost error')
plt.show()