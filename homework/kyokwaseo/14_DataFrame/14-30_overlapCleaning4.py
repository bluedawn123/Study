import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6, 7, 7, 7, 8, 9, 9],
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b", "d", "d", "c", "b", "c", "c"]})

print(dupli_data)
print("--------------")
z = dupli_data.drop_duplicates()
print(z)

'''
    col1 col2
0      1    a
1      1    b
2      2    b
3      3    b
4      4    c
5      4    c
6      6    b
7      6    b
8      7    d
9      7    d
10     7    c
11     8    b
12     9    c
13     9    c
--------------
    col1 col2
0      1    a
1      1    b
2      2    b
3      3    b
4      4    c
6      6    b
8      7    d
10     7    c
11     8    b
12     9    c
'''