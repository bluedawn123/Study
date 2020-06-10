from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastrs = ['6/1/2020', '6/3/2020', '6/4/2020', '6/8/2020', '6/10/2020']
dates = pd.to_datetime(datastrs)
print(dates)
print("=====================")

ts = Series([1, np.nan, np.nan, 8, 10], index = dates)
print(ts)

ts_intp_linear = ts.interpolate()  #선형으로 자동으로 보간이 된다. 
print(ts_intp_linear)
