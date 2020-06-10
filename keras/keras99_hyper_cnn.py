#97번을 98번으로 변경 + score 넣기
# gridsearch, randomsearch 사용
# Dense 모델
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import Dense, MaxPooling2D
import numpy as np
from sklearn.metrics import accuracy_score

''' 1. 데이터 '''

# data 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # 60000, 28, 28
print(x_test.shape)     # 10000, 28, 28

# data 전처리, 리쉐이프
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float')/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float')/255
    # >> 0~255까지 들어가 있는 것을 255으로 나누면 minmax와 같다
    # >> CNN 용

print(x_train.shape)    # (60000, 28, 28, 1)
print(x_test.shape)     # (10000, 28, 28, 1)


# y 원핫인코딩 (차원 확인할 것)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)    # 60000, 10
print(y_test.shape)     # 10000, 10


''' 2. 모델 '''

# GridSearch를 사용할 것이다 그것을 쓰기 위해서 함수제작
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)
#   >> 모델이 들어가는 진짜 함수를 만들 것이다
#gridsearchSV에 들어가는 곳에는 모델이 처음 들어가는데 그것을 만들어줘야한다. 
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (28, 28, 1), name = 'input')
    x = Conv2D(filters = 32, kernel_size = (3, 3),
               padding = 'same', activation = 'relu')(inputs)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters = 32, kernel_size = (3, 3),
               padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(16, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['accuracy'],
                  loss = 'categorical_crossentropy')
    return model




#   >> 함수형 모델과 동일하고 그걸 단지 함수로 감싼 다음에 함수에 들어갈 매개변수 drop, optimizer를 넣어놓고 return 시킨 것이다
#   >> for문 돌리면 2~3개도 쓸 수 있다. (def 윗줄에 for문 들어가면)
#   >> fit은 어디에 ? : grids or randoms에서

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
           "drop" : dropout}
#   >> grids 에 들어갈 요소들이 준비 되었다
#   >> keras에 그냥 사용해서는 안되고 keras의 sklearn의 wrapers class를 땡겨온다

from keras.wrappers.scikit_learn import KerasClassifier  #싸이킷런에 쓸수있게 wrapping을 했다. 
# keras건 sklearn 이건 분류와 회귀가 있다는 점 잊지말자
model = KerasClassifier(build_fn=build_model, verbose=1)


#kerasclassifier에 첫 모델이 들어간다. 

hyperparameters = create_hyperparameters()

search = RandomizedSearchCV(model, hyperparameters, cv=3)
# estimator) sklearn에 쓸 수 있게 wraping을 한 것이다 (위 build_model def와 model 명시 확인)
search.fit(x_train, y_train)

print(search.best_params_)

print("-------------------------------------------------------")

y_pred = model.predict(x_test)

print("최종 정답률 : ", accuracy_score(y_test, y_pred))

print("최적의 매개변수 : ", model.best_estimator_)


'''라고 해도 된다. 
acc =  search.fit(x_test, y_test, verbose = 0)

print("acc : ", acc )
'''


