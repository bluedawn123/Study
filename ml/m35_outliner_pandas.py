import pandas as pd
import numpy as np

# pandas
def outliers(a1):
        quartile_1 = a1.quantile(.25)
        quartile_3 = a1.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((a1 > upper_bound) | (a1 < lower_bound))
         
a3 = pd.DataFrame({'a' : [1, 3, 5, 200, 100, 8],
                   'b' : [300, 100, 6, 8, 2, 3]})

print(77.0 * 1.5)

b3 = outliers(a3)
print(b3)