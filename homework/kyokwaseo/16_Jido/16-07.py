#그냥 예시이다.
from sklearn.linear_model import LogisticRegression

# 모델
model = LogisticRegression()

# 모델학습
# train_data_detail은 데이터의 카테고리 예측사용
# train_data_label는 데이터가 속하는 클래스의 라벨  #뭔소린지?
model.fit(train_data_detail, train_data_label)

# 모델예측
model.predict(data_detail)

#모델의 예측 결과의 정확도
model.score(data_detail, data_true_label)