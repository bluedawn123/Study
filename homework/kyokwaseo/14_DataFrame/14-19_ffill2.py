#nanㅡ을 앞에 있는 데이터로 채워라

import numpy as np
from numpy import nan as NA
import pandas as pd
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

print(sample_data_frame)
print("-------")

# 여기에 해답을 기술하세요
z = sample_data_frame.fillna(method="ffill")

print(z)