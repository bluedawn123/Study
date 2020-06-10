import pandas as pd

names = ['junho', 'jessica', 'john']
births = [123, 223, 223]
custom = [1,3,7]

BabyDataSet = list(zip(names,births))
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])

df.head()