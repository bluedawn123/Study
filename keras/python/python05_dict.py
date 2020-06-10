#3. 딕셔너리 #중복 x
# {키 : 벨류}
# {key : value}

a = {1: 'hi', 2:'hello'}


print(a)
print(a[1])
print(a[2])
print(a[1],     a[2])

b = {'hi' : 3, "dsfsdf" : 4}
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

print("  ")
print(b['hi'], b["dsfsdf"])


#딕셔너리 요소 삭제
print("print(a)를 했을 경우", (a))
del a[1]
print(a)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

z = {1: 'a', 1:'b', 1:'c'}
print(z)                       #출력결과는 {1: 'c'} 마지막 것만 출력되기 때문.

x = {1:'b', 2:'b', 3:'b'}      
print(x)                        #다 출력된다.

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡeeㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())         #키 값인 name, phone, birth만
print(a.values())       #가치값인, yun, 010, 0511만
print(type(a))          #class 'dict'
print(a.get('name'))    #a의이름을 get 하므로, yun
print(a['name'])        #역시 yun
print(a.get('phone'))   #a의 phone을 get하므로, 010
print(a['phone'])       #         동일





















