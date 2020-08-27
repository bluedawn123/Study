#성적 분류
score_list = [90, 50, 65, 85]
number = 1
for score in score_list:
    if score >= 80:
        result = '합격'
    else :
        result = '불합격'
    
    print(" {}번 학생은 {} 입니다.".format(number, result))

    number += 1

#구구단

for firstnumber in range(2, 10):
    for secondnumber in range(1, 10):
        print(firstnumber, "x", secondnumber, "=", firstnumber*secondnumber)
    print(" ")

'''
#랜덤게임
import random

random_number = random.randint(1, 50)  #1에서 50 아무거나 저장
while True:
    num = int(input("숫자를 입력하세요 >> "))
    if num < random_number:
        print("{} 보다 큰 수 입니다. ".format(num))
    elif num > random_number:
        print("{} 보다 작은 수 입니다.".format(num))
    else :
        print(" 정답")
        break
'''

#1부터 100사이의 숫자 중 3의 배수값 들의 합
add = 0

for x in range(1, 100):
    if x % 3 == 0:
        add += x
    print(add)
















