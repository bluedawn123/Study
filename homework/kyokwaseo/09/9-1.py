import numpy as np
import pandas as pd

def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
        df.index = index

        return df


columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

df1 = pd.concat([df_data1, df_data2], axis =0)
df2 = pd.concat([df_data2, df_data2], axis =0)


print("df1 :", df1) 
print("df2 :", df2)