absent = [5, 13]
no_book = [8]

for student in range(1, 31):
    if student in absent:
        continue
    elif student in no_book:
        print("수업끝. {0}은 교무실로와 ".format(student))
        break

    print("{0}, 책을 읽어줘".format(student))


print("-----")
#str -> 길이변환
heroes = ["그루트", "준호", "기범"]
heroes = [len(i) for i in heroes]
print(heroes)  #[3, 2, 2]

#
num = [2, 3, 4]
num1 = [i+100 for i in num]
print(num1)    #[102, 103, 104]      

#퀴즈5 
'''
당신은 택시기사인데 50명의 승객과 매칭 기회가 있다. 총 탑승 승객 수, 총 탑승 시간을 구하는 프로그램을 구하시오.
조건1 : 승객별 운형 소요 시간은 5분 ~ 50분
조건2 : 당신은 소요 시간 5분 ~ 15분 사이만 매칭

출력문예
[o] 1번째 손님 (소요시간 : 15분)
[ ] 2번째 손님 (소요시간 : 50분)
...
[o] 50번째 손님 ( 소요시간 : 14분)

총 탑승 승객 : 2명
'''

from random import *
import random

cnt = 0     #총 탑승승객
alltime = 0 #총 탑승 시간

for i in range(1, 51):       #1~50이라는 수의 승객
    time = randrange(5, 51)  #5분 ~ 50분 소요시간

    if 5 <= time <= 15:      #5분 ~ 15분 사이에 손님이 탄 경우. 이때는 탑승 승객 수를 증가 시켜야한다. 
        print("[o] {0}번째 손님 (소요시간 : {1}분)".format(i, time))
        cnt += 1
        alltime += time
    
    else:                    #매칭 실패
        print("[ ] {0}번째 손님 (소요시간 : {1}분)".format(i, time))

print("총 탑승 승객수 : ", cnt)
print("총 탑승 시간 : ", alltime)




