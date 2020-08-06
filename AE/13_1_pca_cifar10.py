from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
from sklearn.decomposition import PCA

(x), (y) = cifar10.load_data()

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)  #cumsum 누적계산
print(cumsum)

aaa = np.argmax(cumsum >= 0.94)+1            #먼소리?
                                             #0.9479가 7번째인데, 인덱스 순이면 6번째 이므로 + 1

print(cumsum >= 0.94)
print(aaa)

