from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, LGBMRegressor
import pickle

## data
x, y = load_iris(return_X_y = True)
print(x.shape)              # (150, 4)
print(y.shape)              # (150,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)
print(x_train.shape)        # (120, 4)
print(x_test.shape)         # (30, 4)
print(y_train.shape)        # (120,)
print(y_test.shape)         # (30,)

## 모델_1
model = LGBMClassifier(n_jobs = -1,
                       learning_rate = 0.01,
                       max_depth = -1,
                       n_estimators = 500,
                       objective = 'multiclass',
                       metric = ['multi_logloss', 'multi_error'])

## 훈련 예측
model.fit(x_train, y_train,
          verbose = True,
          eval_set = [(x_train, y_train),
                      (x_test, y_test)],
          early_stopping_rounds = 100)

results = model.evals_result_
print("Evaluate's Result : ", results)

## SFM 모델링
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for i in thresholds:
    select = SelectFromModel(estimator = model,
                             threshold = i,
                             prefit = True)
    
    select_x_train = select.transform(x_train)
    select_x_test = select.transform(x_test)

    select_model = LGBMClassifier(n_jobs = -1,
                                  learning_rate = 0.01,
                                  max_depth = -1,
                                  n_estimators = 500,
                                  objective = 'multiclass',
                                  metric = ['multi_logloss', 'multi_error'])
    
    select_model.fit(select_x_train, y_train,
                     verbose = False,
                     eval_set = [(select_x_train, y_train),
                                 (select_x_test, y_test)],
                     early_stopping_rounds = 100)
    
    y_pred = select_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Thresh = %.3f, n = %d, ACC = %.2f%%"
          %(i, select_x_train.shape[1], acc * 100.0))

pickle.dump(select_model, open("./model/LGBM_save/iris_ACC = 93.33%.dat", "wb"))