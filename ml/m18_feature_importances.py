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

import matplotlib.pyplot as plt
import numpy as np



# 1. 데이터
cancer = load_breast_cancer()

x_train,x_test,y_train,y_test = train_test_split( 
    cancer.data, cancer.target ,random_state = 44, shuffle=True,
    train_size=0.8
    )

print("cancer.data 의 shape : ", cancer.data.shape)     # (569, 30)
print("cancer.target의 shape : ", cancer.target.shape)  # (569,)



model = DecisionTreeClassifier(max_depth = 4)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print("모델의 피쳐 임포턴스 : ", model.feature_importances_)

print("cancer.data : ", cancer.data)  #  [[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]

n_features = cancer.data.shape[1]   #30 즉, n feature을 암 데이터의 열의 갯수로

#트리구조
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]         #30
    plt.barh(np.arange(n_features), model.feature_importances_,     #수평 가로 막대를 그린다. (                 )
            align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)         #(y행에 나타나는 것들의 갯수, y행에 나타나는 것들의 이름)

    plt.xlabel("Feature Importacsssnes")                            #가로축
    plt.ylabel("Featuresss")                                        #세로축
    plt.ylim(-1, n_features)                                        #y축의 범위.

plot_feature_importances_cancer(model)
plt.show()
