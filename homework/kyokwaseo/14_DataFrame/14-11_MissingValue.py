import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))
#뭐한지는 모르겠는데 일단 sdf생성

print(sample_data_frame) #아직까진 누락데이터 없다.


#데이터 누락
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 

print("_________________")
print(sample_data_frame)   #누락데이터 생성
