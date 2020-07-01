from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
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

jeju_df = jeju_df.drop(['CARD_CCG_NM', 'HOM_CCG_NM'], axis=1)  #시군구 필요없으니 제거

#남아있는 칼럼들
#CARD_SIDO_NM    카드이용지역_시도 (가맹점 주소 기준)
#STD_CLSS_NM     업종명
#HOM_SIDO_NM     거주지역_시도 (고객 집주소 기준)
#AGE             연령대
#SEX_CTGO_CD     성별 (1: 남성, 2: 여성)
#FLC             가구생애주기 (1: 1인가구, 2: 영유아자녀가구, 3: 중고생자녀가구, 4: 성인자녀가구, 5: 노년가구)
#CSTMR_CNT       이용고객수 (명)
#AMT             이용금액 (원)
#CNT             이용건수 (건)
#year            년
#month           월


















