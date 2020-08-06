import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)  #cumsum 누적계산
print(cumsum)

aaa = np.argmax(cumsum >= 0.94)+1            #먼소리?
                                             #0.9479가 7번째인데, 인덱스 순이면 6번째 이므로 + 1

print(cumsum >= 0.94)
print(aaa)


