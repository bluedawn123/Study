import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split2(dataset, time_steps, y_columns):
    x, y = [], []

    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_columns
        #if x_end_number > 9:
        if y_end_number > len(dataset):
            break

        tmp_x = dataset[i:x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split2(dataset, 4, 2)
print(x)
print("--------------")
print(y)