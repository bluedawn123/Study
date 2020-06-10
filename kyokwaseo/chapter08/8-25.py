#DataFrame생성
import pandas as pd


data = {"fruits" : ["apple", "orange", "banana", "strberry", "kiwi"],
        "year"   : [2001, 2002, 2001, 2008, 2006],
        "time"   : [1, 4, 5, 6, 3]

z = pd.DataFrame(data)
print(z)













