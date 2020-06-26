import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6], 
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b"]}) 

print(dupli_data)
'''
   col1 col2
0     1    a
1     1    b
2     2    b
3     3    b
4     4    c
5     4    c
6     6    b
7     6    b
'''