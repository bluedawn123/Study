import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/samsung.csv', index_col = 0, header =0, sep = ',', encoding = 'CP949')
                                            #인덱스(열 기준)와 헤더(행 기준)는 데이터가 아니다.  
                                            # #header = 0의 의미  ->첫 행을 데이터로 넣지 않겠다.     
hite = pd.read_csv('./data/csv/hite.csv', index_col = 0, header = 0, sep = ',', encoding = 'CP949')




#print(samsung.shape)  #(700, 1)

#None제거 1
samsung = samsung.dropna(axis = 0)  #axis = 0 행(디폴트), axis = 1 열
                                    #이걸 하면 삼성의 nan값이 다 제거된다!  (509, 1)


hite = hite.fillna(method = 'bfill') #위의 결측값을 bfill의 방법(back fill)로 채우겠다. ffill인 경우, 위에값이 아래로 내려온다.
hite = hite.dropna(axis = 0)         #아래의 nan값을 지운다.

#결측치 제거는 다 지우던가 빈 곳을 채워 넣는 방법이 있다.


'''방법 1을 쓰겠다. 
#none제거 방법2
hite = hite[0:509]
#hite.iloc[0, 1:5] = [10, 20, 30, 40]    #iloc = index location              방법이 2가지가 있다. 
hite.loc['2020-06-02', '고가' : '거래량'] = ['100', '200', '300', '400']
'''

#삼성과 하이트의 정렬을 오름차순으로
samsung = samsung.sort_values(['일자'], ascending = ['True'])
hite = hite.sort_values(['일자'], ascending = ['True'])

print(samsung)
print(hite)
print("samsung의 형태 : ", samsung.shape)   #(509, 1)
print("hite의 형태 : ", hite.shape)         #(509, 5)

#콤마제거, 문자를 정수로 형변환
#삼성 콤마제거 후 인트형 변형
for i in range(len(samsung.index)):  
    samsung.iloc[i, 0] = int(samsung.iloc[i, 0].replace(',', ''))  #콤마제거. 37,000을 37000으로 변경

print(samsung)
print(type(samsung.iloc[0,0]))  #정수형으로 변환되었다. 


#하이트 콤마제거 후 인트형 변형
for i in range(len(hite.index)):  
    for j in range(len(hite.iloc[i])):
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',', ''))  #콤마제거. 37,000을 37000으로 변경


print(hite)
print(type(hite.iloc[1,1]))  #정수형으로 변환되었다. 


print("samsung의 형태 : ", samsung.shape)   #(509, 1)
print("hite의 형태 : ", hite.shape)         #(509, 5)

############넘파이 변환 #########################################
samsung = samsung.values
hite = hite.values

print(type(hite))  #ndarray.. 즉 넘파이형 완료

np.save('./data/samsung1.npy', arr = samsung)
np.save('./data/hite1.npy', arr = hite)






































 





































