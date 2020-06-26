import numpy as np
'''
datas = np.array([[1,2,3,14], [11,12,13,14], [21,22,23,24]])

print(datas)
print("dataset.shape : ", datas.shape)
datasT = np.transpose(datas)
print(datasT)
print("dataset.T.shape : ", datasT.shape)
print("-----------위는 심심해서--------")
'''

dataset = np.array([[1,2,3,4,5,6,7,8,9,10], 
                    [11,12,13,14,15,16,17,18,19,20], 
                    [21,22,23,24,25,26,27,28,29,30]])
print(dataset)
print("dataset.shape : ", dataset.shape)  # (3, 10)
datasetT = np.transpose(dataset)
print(datasetT)
'''
[[ 1 11 21]
 [ 2 12 22]
 [ 3 13 23]
 [ 4 14 24]
 [ 5 15 25]
 [ 6 16 26]
 [ 7 17 27]
 [ 8 18 28]
 [ 9 19 29]
 [10 20 30]]
 '''

'''
print(datasetT[0:3, :-1])
[[ 1 11]
 [ 2 12]
 [ 3 13]]
'''

print("datasetT.형태 : ", datasetT.shape)  # (10, 3)
print("len(datasetT)) : ", len(datasetT)) #10


def split3(datasetT, time_steps, y_column):
    x= []
    y= []

    for i in range (len(datasetT)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1

        if y_end_number > len(datasetT):
            break
        
        tmp_x = datasetT[i : x_end_number, : -1]
        tmp_y = datasetT[x_end_number -1 : y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)

x, y = split3(datasetT, 3, 1)
print("x : ",x)
print("x.shape : ", x.shape)  #(8, 3, 2)
print("---------")
print("y : ",y)
print("y.shape : ", y.shape)  #(8, 1)

newY = y.reshape(y.shape[0])
print("newY.shape : ", newY.shape)
print(newY)
