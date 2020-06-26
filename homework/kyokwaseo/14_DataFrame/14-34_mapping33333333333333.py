#새로운 컬럼 regin을 추가. 데이터 없으면 nan으로

import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", 
                        "서울", "부산", "대전", "광주", "서울"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["영식", "순돌", "짱구", "태양", 
                        "션", "유리", "현아", "태식", "민수", "호식"]}
attri_data_frame1 = DataFrame(attri_data1)

print(attri_data_frame1)

city_map ={"서울":"서울", 
           "광주":"전라도", 
           "부산":"경상도", 
           "대전":"충정도"}

print(city_map)

attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)

#region으로 열추가. 

print(attri_data_frame1)

'''
    ID city  birth_year name region
0  100   서울        1990   영식     서울
1  101   부산        1989   순돌    경상도
2  102   대전        1992   짱구    충정도
3  103   광주        1997   태양    전라도
4  104   서울        1982    션     서울
5  106   서울        1991   유리     서울
6  108   부산        1988   현아    경상도
7  110   대전        1990   태식    충정도
8  111   광주        1995   민수    전라도
9  113   서울        1981   호식     서울
'''

#어렵다.