'''
표준 체중을 구하는 프로그램을 작성하시오

표준 체중 : 각 개인의 키에 적당한 체중

(성별에 따른 공식)
남 : 키 x 키 x 22
여 : 키 x 키 x 21

조건 1 : 표준 체중은 별도의 함수 내에서 계산
        * 함수명 : std_weight
        * 전달값 : 키, 성별

조건 2 : 표준 체중은 소수점 둘째자리 까지 표시

출력예제 : 키 175남자의 표준 체중은 67,38kg 입니다. 
'''

def std_weight(height, gender):  #키는 m 단위, 성별은 "남자", "여자"
    if gender == "남자":
        return height* height * 22
    
    else:
        return height*height*21

height = 183
gender = "남자"
weight = round(std_weight(height / 100, gender), 2)
print("키 {0}cm {1}의 표준 체중은 {2}kg 입니다.".format(height, gender, weight))