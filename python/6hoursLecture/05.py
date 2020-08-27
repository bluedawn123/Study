#시험 성적
scores = {"수학 " : 0, "영어 " : 30, "코딩 " : 95}

for subject, score in scores.items():
    print(subject.ljust(8), str(score).rjust(4), sep = ":")  
    
    #왼쪽으로 정렬을 하는데 총 8칸 공간을 확보한 상태에서 정렬

print("--------------------")
for num in range(1, 21):
    print("대기번호 : " + str(num).zfill(3))  #3크기만큼에 0으로 채우겠다.

print("--------------------")
answer = input("아무거나 입력해라 : ")
print("입력하신 값은, " + answer + " 입니다")