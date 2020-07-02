from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression , Ridge , Lasso

#%matplotlib inline

#1. 데이터 확인
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

jeju_df = pd.read_csv('./data/csv/jeju/201901-202003.csv')
print(jeju_df.shape)   #(24697792, 12)
print(jeju_df.info())

#REG_YYMM        년월
#CARD_SIDO_NM    카드이용지역_시도 (가맹점 주소 기준)
#CARD_CCG_NM     카드이용지역_시군구 (가맹점 주소 기준)
#STD_CLSS_NM     업종명
#HOM_SIDO_NM     거주지역_시도 (고객 집주소 기준)
#HOM_CCG_NM      거주지역_시군구 (고객 집주소 기준)
#AGE             연령대
#SEX_CTGO_CD     성별 (1: 남성, 2: 여성)
#FLC             가구생애주기 (1: 1인가구, 2: 영유아자녀가구, 3: 중고생자녀가구, 4: 성인자녀가구, 5: 노년가구)
#CSTMR_CNT       이용고객수 (명)
#AMT             이용금액 (원)
#CNT             이용건수 (건)
'''
#null 값 확인했으나 시군구는 시도별에 속하므로 그냥 드롭하거나 합쳐버린다. 
print('데이터 세트 null 값 개수 ', jeju_df.isnull().sum().sum())  #데이터 세트 null 값 개수  235000 
print(jeju_df.isnull().sum())

CARD_CCG_NM      87213        카드이용지역_시군구 = 87213  null값.
HOM_CCG_NM      147787        거주지역_시군구     = 147787 null값.  
'''
jeju_df = jeju_df.fillna('')


#2. 날짜 (년,월 분리)
def grap_year(data):
    data = str(data)
    return int(data[:4])

def grap_month(data):
    data = str(data)
    return int(data[4:])

jeju_df['year'] = jeju_df['REG_YYMM'].apply(lambda x: grap_year(x))
jeju_df['month'] = jeju_df['REG_YYMM'].apply(lambda x: grap_month(x))
jeju_df = jeju_df.drop(['REG_YYMM'], axis=1)

#년, 월을 분리시키고 원래 있던 REG_YYMM 삭제

jeju_df = jeju_df.drop(['CARD_CCG_NM', 'HOM_CCG_NM','FLC' ], axis=1, inplace=True)  #시군구, FLC 필요없으니 제거

#남아있는 칼럼들
#CARD_SIDO_NM    카드이용지역_시도 (가맹점 주소 기준)
#STD_CLSS_NM     업종명
#HOM_SIDO_NM     거주지역_시도 (고객 집주소 기준)
#AGE             연령대
#SEX_CTGO_CD     성별 (1: 남성, 2: 여성)
#CSTMR_CNT       이용고객수 (명)
#AMT             이용금액 (원)
#CNT             이용건수 (건)
#year            년
#month           월

from sklearn.metrics import mean_squared_error, mean_absolute_error

# log 값 변환 시 NaN등의 이슈로 log() 가 아닌 log1p() 를 이용하여 RMSLE 계산
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# 사이킷런의 mean_square_error() 를 이용하여 RMSE 계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))

# MSE, RMSE, RMSLE 를 모두 계산 
def evaluate_regr(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    # MAE 는 scikit learn의 mean_absolute_error() 로 계산
    mae_val = mean_absolute_error(y,pred)
    print('RMSLE: {0:.3f}, RMSE: {1:.3F}, MAE: {2:.3F}'.format(rmsle_val, rmse_val, mae_val))

'''
#일단 타겟값 설정
y_target = jeju_df['AMT']
X_features = jeju_df.drop(['AMT'], axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test ,pred)

print(evaluate_regr)
'''