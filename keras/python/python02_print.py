#print문과 format함수
a = '사과'
b = '배'
c = '옥수수'

print('준호는 잘생겼다.')

print(a)
print(a,     b)
print(a + b)
print("나는 {0}을 먹었다". format(a))   #중괄호안에 format(a)를 넣겠다. 중괄호 0 자리에다가. 왜냐면 파이썬의 순서는 0이 첫번째니깐.
print("나는 {0}와 {1}을 먹었다.". format(a, b))  #중괄호에 0, 1에 a,b가 들어간다. 
print("나는{0}와 {1} {2}를 먹었다". format(a, b, c)) #역시 같다. 생략. 


print('나는 ', a,'를 먹었다.', sep='')
print('나는 ',a, '와 ' , b,'를 먹었다.', sep='')  #sep = seperate의미. 따옴표 사이에 무엇을 넣을 것인가?
print('나는 ',a, '와 ' , b,'를 먹었다.', sep='#')  