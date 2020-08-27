def open_account():
    print("새로운 계좌 생성되었습니다. ")

def deposit(balance, money):
    print("입금이 완료 되었습니다. 잔액은 {0} 원 입니다. ".format(balance + money))
    return balance + money

#def withdraw(balance, money)

def withdraw(balance, money):
    if balance >= money:
        print("출금이 완료되었습니다. 잔액은, {0}원 입니다. ".format(balance - money))
        return balance - money
    
    else:
        print("출금이 완료되지 않았습니다. 잔액은 {0}원 입니다. ".format(balance))
        return balance

def withdraw_night(balance, money):
    commission = 100
    return commission, balance - money - commission

open_account()

balance = 1000  #잔액
balance = deposit(balance, 2000)

commission, balance = withdraw_night(balance, 500)

print("수수료는 {0}원이며, 잔액은 {1}원 입니다.".format(commission, balance))

