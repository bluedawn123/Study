import matplotlib
from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt


# 데이터를 생성합니다
# 이 데이터는 선형 분리가 가능하지 않기 때문에, 다른 데이터를 준비합니다
data, label = make_gaussian_quantiles(n_samples=1000, n_classes=2, n_features=2, random_state=42)

# 모델을 구축합니다
# LinearSVC 대신 SVC를 사용합니다
model = SVC()

# 모델을 학습합니다
model.fit(data, label)

# 정확도를 산출합니다
print(model.score(data, label))