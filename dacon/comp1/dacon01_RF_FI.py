from sklearn.datasets import load_breast_cancer
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import missingno as msno
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor        

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col=0)

x_test1 = test
y_predict = submission

feature_names = train.columns[:-4]

print(train.columns)

print("train.shape : ", train.shape)                 #(10000, 75)      여기서 x_train x_test를 만들어야한다.
print("test.shape : ", test.shape)                   #(10000, 71)      test로 x_pred를 만들어야 한다. 
print("submission.shape : ", submission.shape)       #(10000, 4)                y_pred

#결측치확인
print(train.isnull().sum())  #트레인에서 isnull이 있는 것을 합해서 보여줘
print(list(train.isnull().sum())) 

print(test.isnull().sum()) 
print(list(test.isnull().sum())) 



#결측치 제거 후 확인 
train = train.interpolate().fillna(method = 'bfill').fillna(method = 'ffill') #선형 보간법+뒤, 앞에서 채워넣기
test = test.interpolate().fillna(method = 'bfill').fillna(method = 'ffill')  #선형 보간법.

print(train.isnull().sum())
print(test.isnull().sum())


#train = train.fillna(method = 'bfill')   #뒤에 있는 거 끌어오기
#test = train.fillna(method = 'bfill')    #뒤에 있는 거 끌어오기

#numpy 저장.
train = train.values

np.save('./data/dacon/train.npy', arr=train)  

print("  ")
print("---------이후는 넘파이 변경 후 load해서 이어서 -------------")

train = np.load('./data/dacon/train.npy', allow_pickle=True)

#데이터
x = train[:, :71]     #[:]모든행  [0:72] 0~71까지의 인데스 
y = train[:, 71: ]    #[:]모든행 ,[71:] 71부터 까지의 인데스 

print("슬라이싱 후의 x의 형태 : ", x)
print("슬라이싱 후의 y의 형태 : ", y)

print("x.shape : ", x.shape)  #(10000, 71)
print("y.shape : ", y.shape)  #(10000, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.8)

print("x_train 모양 : ", x_train.shape)        #(8000, 71)         
print("x_test 모양 : ", x_test.shape)          #(2000, 71)     
print("y_train 모양 : ", y_train.shape)        #(8000, 4)       
print("y_test 모양 : ", y_test.shape)          #(2000, 4)

#트리구조
model = RandomForestRegressor() 

model.fit(x_train, y_train)
# acc = model.score(x_test, y_test)

print(model.feature_importances_)


n_features = x.data.shape[1]   #30

def plot_feature_importances_x(model):
    n_features = x.data.shape[1]         #30
    plt.barh(np.arange(n_features), model.feature_importances_,     #수평 가로 막대를 그린다. (                 )
            align = 'center')
    plt.yticks(np.arange(n_features), feature_names)         #(y행에 나타나는 것들의 갯수, y행에 나타나는 것들의 이름)

    plt.xlabel("Feature Importacsssnes")   #가로축
    plt.ylabel("Featuresss")               #세로축
    plt.ylim(-1, n_features)               #y축의 범위.

plot_feature_importances_x(model)
plt.show()
