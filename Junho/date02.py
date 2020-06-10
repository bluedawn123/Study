import datetime

now = datetime.datetime.now()

if now.hour < 12:
    print("오전이다")

if now.hour >= 12:
    print("오후이다.")

print("---------------------------------------------------------------------------")

x = input("지금 시각을 입력하시오")
x = int(x)

if x >= 12:
    print("오후이다")

if x < 12:
    print("오전이다")

print("---------------------------------------------------------------------------")

import datetime

now = datetime.datetime.now()

if 3 <= now.month <= 5:
    print("봄입니다.")

if 6 <= now.month <= 9:
    print("여름입니다.")
