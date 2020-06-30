from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib

cust_df = pd.read_csv('./data/csv/santander/train.csv', encoding='latin-1')
print('dataset shape:', cust_df.shape)
cust_df.head(3)


# var3 피처 값 대체 및 ID 피처 드롭
cust_df['var3'].replace(-999999,2, inplace=True)
cust_df.drop('ID',axis=1 , inplace=True)

# 피처 세트와 레이블 세트분리. 레이블 컬럼은 DataFrame의 맨 마지막에 위치해 컬럼 위치 -1로 분리
X_features = cust_df.iloc[:, :-1]
y_labels = cust_df.iloc[:, -1]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels,
                                                    test_size=0.2, random_state=0)
train_cnt = y_train.count()
test_cnt = y_test.count()

from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(n_estimators=100)

evals = [(X_test, y_test)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=evals,
                verbose=True)

lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV

# 하이퍼 파라미터 테스트의 수행 속도를 향상시키기 위해 n_estimators를 100으로 감소
'''
lgbm_clf = LGBMClassifier(n_estimators=150)

params = {'num_leaves': [24, 32, 64 ],
          'max_depth':[64, 128, 160],
          'min_child_samples':[30, 60, 100],
          'subsample':[0.8, 1]}


# cv는 3으로 지정 
gridcv = GridSearchCV(lgbm_clf, param_grid=params, cv=3)
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc",
           eval_set=[(X_train, y_train), (X_test, y_test)])

print('GridSearchCV 최적 파라미터:', gridcv.best_params_)
lgbm_roc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], average='macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
'''
#lgbm_clf = LGBMClassifier(n_estimators=300, num_leaves=64, sumbsample=0.8, min_child_samples=30,
            #max_depth= 64)
#ROC AUC: 0.8442

#해당 하이퍼 파라미터를 lightGBM에 적용하고 다시 학습해 ROC-AUC 측정결과를 도출


lgbm_clf = LGBMClassifier(n_estimators=500, num_leaves=64, sumbsample=0.8, min_child_samples=30,
                          max_depth= 64)

evals = [(X_test, y_test)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=80, eval_metric="auc", eval_set=evals,
                verbose=True)

lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
