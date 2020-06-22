# XGB도 역시 evaulate가 있다

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

#1. 데이터
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8, random_state=99)

#2. 모델 구성
model = XGBRegressor(n_estimators=1000, learning_rate=0.1)
# n_estimators는 

#3. 훈련
model.fit(x_train, y_train, verbose=True, eval_metric="error", 
            eval_set=[(x_train,y_train), (x_test, y_test)])
# verbose 딥러닝의 metrics가 있었음. 머신러닝의 지표는 rmse, mae, logloss, error(=acc), auc(정확도 acc의 친구)
# error가 0.8이면 acc가 0.2

#4. 평가
result = model.evals_result()
print("evals_result : ", result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
#print("r2 Score : %.2f%%" %(r2 * 100.0))
print("r2 : ", r2)