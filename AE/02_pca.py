import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

print("pca전 x : ", x)
print(" ")

pca = PCA(n_components=10)
x2 = pca.fit_transform((x))
pca_evr = pca.explained_variance_ratio_
print(pca_evr)
print(sum(pca_evr))


