from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#(samples=1000, features=2,random_state=42)
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, random_state=0)

#데이터를 분할하세요(테스트 크기=0.2,random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

#모델 구축
model_list = { '로지스틱 회귀':LogisticRegression(),
              '선형 SVM':LinearSVC(),
              '비선형 SVM':SVC(),
              '결정 트리':DecisionTreeClassifier(),
              '랜덤 포레스트':RandomForestClassifier()}

print(model_list.items())


#베낌
#for 문
for model_name, model in model_list.items():
    # 모델을 학습시킵니다
    model.fit(train_X, train_y)
    print(model_name)
    # 정확도를 출력하세요
    print('정확도: ' + str(model.score(test_X, test_y)))
    print()