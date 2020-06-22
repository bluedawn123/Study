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

# 1. 데이터
cancer = load_breast_cancer()

x_train,x_test,y_train,y_test = train_test_split( 
    cancer.data, cancer.target ,random_state = 66, shuffle=True,
    train_size=0.8
    )


model = DecisionTreeClassifier(max_depth = 4)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)

