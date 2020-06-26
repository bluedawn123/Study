#키별 평균

import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", 
                header = None)
df.columns=[" ", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", 
            "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]


print(df)
'''
        Alcohol  Malic acid  ...   Hue  OD280/OD315 of diluted wines  Proline    
0    1    14.23        1.71  ...  1.04                          3.92     1065    
1    1    13.20        1.78  ...  1.05                          3.40     1050    
2    1    13.16        2.36  ...  1.03                          3.17     1185    
3    1    14.37        1.95  ...  0.86                          3.45     1480    
4    1    13.24        2.59  ...  1.04                          2.93      735    
..  ..      ...         ...  ...   ...                           ...      ...    
173  3    13.71        5.65  ...  0.64                          1.74      740    
174  3    13.40        3.91  ...  0.70                          1.56      750    
175  3    13.27        4.28  ...  0.59                          1.56      835    
176  3    13.17        2.59  ...  0.60                          1.62      840    
177  3    14.13        4.10  ...  0.61                          1.60      560    

[178 rows x 14 columns]
'''
print(df["Alcohol"].mean())  #13.000617977528083

#df[열].mean()  하면 해당 행의 평균값이 나온다. 예를들어,

z = df['Ash'].mean()
print(z)
#이런식으로 해도 된다. 2.3665168539325854