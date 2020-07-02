from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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

# 데이터 정제
df = jeju_df.copy()
df = df.drop(['CARD_CCG_NM', 'HOM_CCG_NM'], axis=1)

columns = ['CARD_SIDO_NM', 'STD_CLSS_NM', 'HOM_SIDO_NM', 'AGE', 'SEX_CTGO_CD', 'FLC', 'year', 'month']
df = df.groupby(columns).sum().reset_index(drop=False)

print(df)
'''
        CARD_SIDO_NM  ... CNT
0                 강원  ...   4        
1                 강원  ...   3        
2                 강원  ...   3        
3                 강원  ...   3        
4                 강원  ...   3        
...              ...  ...  ..
1057389           충북  ...   4        
1057390           충북  ...   7        
1057391           충북  ...   7        
1057392           충북  ...   3        
1057393           충북  ...   3        
'''


# 라벨 인코딩
dtypes = df.dtypes
encoders = {}
for column in df.columns:
    if str(dtypes[column]) == 'object':
        encoder = LabelEncoder()
        encoder.fit(df[column])
        encoders[column] = encoder
        
df_num = df.copy()        
for column in encoders.keys():
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])



####################################################################################
#년,월 나누고 라벨인코딩하고, 쓸데없는 거 지운 상황.
#모델링 시작. 
# log 값 변환 시 NaN등의 이슈로 log() 가 아닌 log1p() 를 이용하여 RMSLE 계산########################
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
#################################################################################################

train_num = df_num.sample(frac=1, random_state=0)
X_features = train_num.drop(['CSTMR_CNT', 'AMT', 'CNT'], axis=1)  #(1057394, 8)   
y_target = np.log1p(train_num['AMT'])                             #(1057394,  )

x_train, x_test, y_train, y_test = train_test_split(
    X_features, y_target, train_size=0.8)

'''
1. LinearRegression()
#lr_reg = LinearRegression()       RMSLE: 0.161, RMSE: 2.492, MAE: 2.018
#lr_reg.fit(x_train, y_train)
#pred = lr_reg.predict(x_test)
'''

#2. RF
model = RandomForestRegressor(n_jobs=-1, random_state=0)   RMSLE: 0.054, RMSE: 0.740, MAE: 0.512 
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("pred : ", pred)
evaluate_regr(y_test, pred)
print("evaluate_regr : ", evaluate_regr)
'''
parameters = [
               {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.5, 0.01, 0.001], "max_depth" : [4, 5, 6]} ,                                                       
               {"n_estimators" : [80, 90, 100, 110], "learning_rate" : [0.1, 0.08, 0.5, 0.001], "max_depth" : [4, 5, 6, 7],
                "colsample_bytree":[0.6, 0.9, 1],"colsample_bylevel" : [0.6, 0.7, 0.8, 0.9]},
               {"n_estimators" : [90, 110], "learning_rate" : [0.1, 0.001, 0.5], "max_depth" : [4, 5, 6], 
               "colsample_bytree":[0.6,0.9,1],"colsample_bylevel" : [0.6, 0.7, 0.8]}
             ]

n_jobs = -1   

#추후 CV꼭 쓰고 Feature_importance도 써야한다.

model = GridSearchCV(LGBMRegressor(), parameters, cv=5, n_jobs=-1)          # RMSLE: 0.064

model.fit(x_train, y_train)
print("------------------------------------")
print(model.best_estimator_)
print(model.best_params_)
print("------------------------------------")
pred = model.predict(x_test)
print("pred : ", pred)
evaluate_regr(y_test, pred)
print("evaluate_regr : ", evaluate_regr)
'''

#예측 템플릿 만들기
CARD_SIDO_NMs = df_num['CARD_SIDO_NM'].unique()
STD_CLSS_NMs  = df_num['STD_CLSS_NM'].unique()
HOM_SIDO_NMs  = df_num['HOM_SIDO_NM'].unique()
AGEs          = df_num['AGE'].unique()
SEX_CTGO_CDs  = df_num['SEX_CTGO_CD'].unique()
FLCs          = df_num['FLC'].unique()
years         = [2020]
months        = [4, 7]

temp = []
for CARD_SIDO_NM in CARD_SIDO_NMs:
    for STD_CLSS_NM in STD_CLSS_NMs:
        for HOM_SIDO_NM in HOM_SIDO_NMs:
            for AGE in AGEs:
                for SEX_CTGO_CD in SEX_CTGO_CDs:
                    for FLC in FLCs:
                        for year in years:
                            for month in months:
                                temp.append([CARD_SIDO_NM, STD_CLSS_NM, HOM_SIDO_NM, AGE, SEX_CTGO_CD, FLC, year, month])
temp = np.array(temp)
temp = pd.DataFrame(data=temp, columns=x_train.columns)


# 예측
pred = model.predict(temp)
pred = np.expm1(pred)
temp['AMT'] = np.round(pred, 0)
temp['REG_YYMM'] = temp['year']*100 + temp['month']
temp = temp[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
temp = temp.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop=False)


# 디코딩 
temp['CARD_SIDO_NM'] = encoders['CARD_SIDO_NM'].inverse_transform(temp['CARD_SIDO_NM'])
temp['STD_CLSS_NM'] = encoders['STD_CLSS_NM'].inverse_transform(temp['STD_CLSS_NM'])


submission = pd.read_csv('data/submission.csv', index_col=0)
submission = submission.drop(['AMT'], axis=1)
submission = submission.merge(temp, left_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission.index.name = 'id'
submission.to_csv('./data/csv/jeju/submission.csv', encoding='utf-8-sig')
submission.head()


"""
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



"""
