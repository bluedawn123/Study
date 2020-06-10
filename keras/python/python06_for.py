a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}


#  for i in 100:   100개에서 하나씩 늘리라

for i in a.keys():  #a.keys a의 keys가 3개(name, phone, birth)
    print(i)

#즉, a.keys 값들을 i에 넣어줘서 출력하겠다는 의미.
#name, phone, birth가 차례대로 출력된다. 

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

for c in a.values():  #a.keys a의 keys가 3개(name, phone, birth)
    print(c)

#즉, a.values 값들을 i에 넣어줘서 출력하겠다는 의미.
#yun, 010, 0511 가 차례대로 출력된다. 

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:  #인자의 갯수만큼 돌려랴. 즉 10번 돌려라. 
    i = i*i
    print(i)

#1,4,9,16,25,36,49,64,81,100로 출력된다. 

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:  #인자의 갯수만큼 더해랴. 즉 10번 돌려라. 
    i = i+i
    print(i)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:  #인자의 갯수만큼 더해랴. 즉 10번 돌려라. 
    i = i+1
    print(i)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("While문")

'''
while 조건문 :     #참일 동안 계속 된다. 
    수행할 문장.
'''

###if문
'''
if 1 :
    print('True')
else :
    print('False')
'''
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

print("If문")
if 3 :
    print('Ture')
else :
    print('False')



if 0 :
    print('True')
else :
    print('False')

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("비교연산자")

a = 1
if a == 1:
    print('출력잘되')
else:
    print('안되')



b = input("숫자를 입력하시오 > ")
b = int(b)
if b == 3:
    print("b는 3이다")
else:
    print("b는 3이 아니다.")


print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("For문 섞어서")

jumsu = [90, 25, 35, 60, 75]
number = 0;

for i in jumsu:
    
    if i >= 60:
        print("졸시 통과")
        number = number + 1


print("합격인원 : ", number, "명")


print("")
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("Break, continue 문 섞어서")


jumsu = [90, 25, 35, 60, 75]
number = 0;
for i in jumsu:
    if i < 30:
        break


    if i >= 60:
        print("졸시 통과")
        number = number + 1


print("합격인원 : ", number, "명")


print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡcontinueㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")


jumsu = [90, 25, 35, 60, 75]
number = 0;
for i in jumsu:
    if i < 60:
        continue


    if i >= 60:
        print("졸시 통과")
        number = number + 1


print("합격인원 : ", number, "명")











