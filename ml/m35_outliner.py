import numpy as np

def outliners(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    
    print("1사 분위 : ", quartile_1)  #3.25
    print("3사 분위 : ", quartile_3)  #97.5

    iqr = quartile_3 - quartile_1

    print("iqr : ", iqr)  #94.25
    
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    print("lower_bound : ", lower_bound)  #-138.125
    print("upper_bound : ", upper_bound)  #238.875

    return np.where((data_out>upper_bound) | (data_out<lower_bound))  


a = np.array([1,2,3,4,1000,6,7,5000, 90, 100])
b = outliners(a)


print("이상치의 위치 : ", b)   #(array([4, 7], dtype=int64),)  즉 1000과 5000이 이상치이다. 
print("a.shape : ", a.shape)