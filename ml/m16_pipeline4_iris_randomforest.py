from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
iris=load_iris()
x=iris.data
y=iris.target


x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.2, shuffle=True, random_state=43)

# 2. 모델

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    parameters = [{"randomforestclassfier__n_jobs" :[1]}]
    return{"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout, "randomforestclassfier" : parameters}


# parameters = [
#     {"randomforestclassfier__n_jobs" :[1]},
#     # {"randomforestclassfier__n_estimator" :[range(1,100,1)]},
#     # {"svc__C" :[1, 10, 100,1000], "svc__kernel" :['rbf'], 'svc__gamma':[0.001, 0.0001]},
#     # {"svc__C" :[1, 10, 100, 1000], "svc__kernel" :['sigmoid'], 'svc__gamma':[0.001, 0.0001]}
# ]



# from keras.wrappers.scikit_learn import KerasClassifier
# model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model, hyperparameters, cv=3, n_jobs=1)
# search.fit(x_train, y_train)

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler",MinMaxScaler()),('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = RandomizedSearchCV(pipe, hyperparameters, cv=5)




pipe.fit(x_train, y_train)


print("acc : ", pipe.score(x_test,y_test))