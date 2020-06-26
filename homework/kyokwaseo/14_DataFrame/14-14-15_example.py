#0,2열은 남기고 NaN포함 행 삭제

import numpy as np
from numpy import nan as NA
import pandas as pd
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

sample_data_frame[[0, 2]].dropna()

print(sample_data_frame)  #결측치가 있는 걸 확인할 수 있다.

print(" ")
z = sample_data_frame[[0, 2]].dropna() #(0,2)행을 남기고 결측치 있는 것 다 삭제
print(z)