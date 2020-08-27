#지역변수, 권역변수

gun = 10

def checkpoint(soldier):
    global gun #전역 공간에 있는 gun 사용
    gun = gun - soldier
    print("[함수 내] 남은 총 : {0}".format(gun))


def checkpoint_return(gun, solider):
    gun =  gun - solider
    print("[함수 내] 남은 총 : {0}".format(gun))
    return gun


print("-------전역 공간의 gun 사용 경우------")
print("전체 총의 수 : {0}".format(gun))
checkpoint(2) #군인 2명
print("남은 총 : {0}".format(gun))   

print("  ")

gun = checkpoint_return(gun, 3)
print("남은 총의 갯수 : ", gun)
