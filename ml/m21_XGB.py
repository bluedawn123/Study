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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier


# 1. 데이터
cancer = load_breast_cancer()

x_train,x_test,y_train,y_test = train_test_split( 
    cancer.data, cancer.target ,random_state = 44, shuffle=True,
    train_size=0.8
    )

print("cancer.data 의 shape : ", cancer.data.shape)     # (569, 30)
print("cancer.target의 shape : ", cancer.target.shape)  # (569,)


#model = RandomForestClassifier()                        #max_features : 기본값을 써라. 
                                                        #n_estimatiors : 클수록좋으나 단점, 메모리 차지. 기본값은 100
                                                        #n_jobs = -1, 병렬처리    

#model = GradientBoostingClassifier()
model = XGBClassifier()

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)

n_features = cancer.data.shape[1]   #30




def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]         #30
    plt.barh(np.arange(n_features), model.feature_importances_,             #수평 가로 막대를 그린다. (                 )
            align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)                 #(y행에 나타나는 것들의 갯수, y행에 나타나는 것들의 이름)

    plt.xlabel("Feature Importacsssnes")                            #가로축
    plt.ylabel("Featuresss")                                        #세로축
    plt.ylim(-1, n_features)                                        #y축의 범위.

plot_feature_importances_cancer(model)
plt.show()



