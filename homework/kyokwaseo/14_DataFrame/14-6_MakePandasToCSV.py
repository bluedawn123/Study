import pandas as pd


data = {
        "city": ["Nagano", "Sydney", "Salt Lake City", "Athens", "Torino", "Beijing",
         "Vancouver", "London", "Sochi", "Rio de Janeiro"],
       
        "year": [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016],
       
        "season": ["winter", "summer", "winter", "summer", "winter", 
       
        "summer", "winter", "summer", "winter", "summer"]
        
        }

df = pd.DataFrame(data)

print(df)

df.to_csv("csv12345678910.csv")  #☆☆☆ 판다스를 csv로 할때는 to_csv

#왜 안되는지 모르겠네...