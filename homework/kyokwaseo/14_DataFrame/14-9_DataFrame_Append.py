#adf1에 adf2행을 추가 후 출력.
#아이디 오름차순, 행번호도 오름차순

import pandas as pd
from pandas import Series, DataFrame

attri_data1 = {
                "ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
                "city": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo", "Tokyo", "Osaka", "Kyoto",
                "Hokkaido", "Tokyo"],
                "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
                "name": ["Hiroshi", "Akiko", "Yuki", "Satoru", "Steeve", "Mituru", "Aoi", "Tarou", 
                "Suguru", "Mitsuo"]
                }
attri_data_frame1 = DataFrame(attri_data1) #attri_data1을 데이터프레임으로 하고 그것을 a_d_f1으로 저장.

print(attri_data1)
print("-------------------")


attri_data2 = {"ID": ["107", "109"],
               "city": ["Sendai", "Nagoya"],
               "birth_year": [1994, 1988]}
attri_data_frame2 = DataFrame(attri_data2)  #attri_data2을 데이터프레임으로 하고 그것을 a_d_f2으로 저장.

print(attri_data2)
print("----------------")

z = attri_data_frame1.append(attri_data_frame2).sort_values(by="ID", ascending=True).reset_index(drop=True)


#at1을 붙인다 at2와.. 값들을 정렬한다(아이디 오름) ????

print("결합")
print(z)