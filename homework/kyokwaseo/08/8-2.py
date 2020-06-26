import pandas as pd

data = {"fruits " : ["apple", "orange", "banana", "strayberry", "kiwifruit"],
        "year"    : [2001, 2002, 2001, 2008, 2006],
        "time"    : [1, 4, 5, 6, 3]
}


a = pd.DataFrame(data)
print(a)