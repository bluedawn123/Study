import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

#1. 데이터 불러오기 및 파악
picher_file_path = './data/bunsukcsv/picher_stats_2017.csv'
picher = pd.read_csv(picher_file_path)
print(picher.columns)
'''
Index(['선수명', '팀명', '승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9',
       '볼넷/9', '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR',
       '연봉(2018)', '연봉(2017)'],
      dtype='object')
      '''
print(picher.head())

'''
  선수명   팀명   승   패  세  홀드  블론  경기  선발  ...  BABIP  LOB%   ERA  RA9-WAR 
 연봉(2018)  연봉(2017)
0   켈리   SK  16   7  0   0   0  30  30  ...  0.342  73.7  3.60     6.91  3.69        
85000
1   소사   LG  11  11  1   0   0  30  29  ...  0.319  67.1  3.88     6.80  3.52          
50000
2  양현종  KIA  20   6  0   0   0  31  31  ...  0.332  72.1  3.44     6.54  3.94        
150000
3  차우찬   LG  10   7  0   0   0  28  28  ...  0.298  75.0  3.43     6.11  4.20        
100000
4  레일리   롯데  13   7  0   0   0  30  30  ...  0.323  74.1  3.80     6.13  4.36  

[5 rows x 22 columns]
'''
print("picher.shape : ", picher.shape) 
#중요한 피쳐 5개 : 'FIP', 'WAR', '볼넷/9', '삼진/9', '연봉(2017)'
