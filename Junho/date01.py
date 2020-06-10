#날짜/시간 활용하기
import datetime

now = datetime.datetime.now()  #내장함수

print(now.year,"   년")
print(now.hour,"시")

print("---------------------------------------------------------------------------")
#format 결합

b = datetime.datetime.now()

print("{}년 {}월 {}일 {}시 {}분 {}초".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
 
)