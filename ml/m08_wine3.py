import pandas as pd
import matplotlib.pyplot as plt

#와인데이터 소환
wine = pd.read_csv("./data/csv/winequality-white.csv", sep = ';', header = 0)

#판다스 안의 계수를 그룹별로 정리
count_data = wine.groupby('quality')['quality'].count()   #퀄리티 안의 개체별로 숫자를 센다. 

print(count_data)
''' 즉, 퀄리티안에 있는 개체별로 숫자를 세어준다.
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
count_data.plot()
plt.show()

#슬라이싱
y = wine['quality']
x = wine.drop('quality', axis = 1)

print("x의 모양 : ", x)
print("y의 모양 : ", y)
print("x.shape의 모양 : ", x.shape)
print("y.shape의 모양 : ", y.shape)

#y레이블 축소. 레이블의 개념?  좋음, 보통, 아주좋음...이런식으로 포문을 써서 분류
newlist = []

for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else :
        newlist += [2]

y = newlist


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, x_test)
acc = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_pred))
print("acc       : ", acc)






























