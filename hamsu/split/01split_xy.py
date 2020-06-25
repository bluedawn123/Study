import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_01(dataset, time_steps):
    x = list()
    y = list()

    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) - 1:
            break
        
        tmp_x = dataset[i:end_number] 
        tmp_y = dataset[end_number]

        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_01(dataset, 4)

print(x, y)
            