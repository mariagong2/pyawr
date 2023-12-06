"Page 159"
'''
x = int (input("입력>>"))
under_20 = (x<20)
if under_20:
    print(x,"는 20보다 작다! ")
'''
# if조건문은 뒤에 클론:을 반드시 사용해야합니다

#조건문의 기본 사용
'''
while True: # 무한루프를 뜻함 
    user_input = input("숫자를 입력하세요 ('종료' 입력 시 프로그램 종료): ")

    # '종료'를 입력하면 프로그램 종료
    if user_input.lower() == '종료':
        print("프로그램을 종료합니다.")
        break

    try:
        number = int(user_input)

        if number < 0:
            print("음수를 쓰지 말랬지")
        elif number > 0:
            print("양수입니다")
        elif number == 0:
            print("아마 0을 넣었던거같다")

    except ValueError:
        print("문자는 사용 금지입니다.")
        break
    
'''


'''
#날짜/시간 출력하기 
import datetime

now =datetime.datetime.now()
print(now.year,"년도")
print(now.month,"월")
print(now.day,"데이")
print(now.hour,"시간")
print(now.minute,"분")
print(now.second,"초")
'''
'''
while True:
    number = input("숫자를 입력하세요>")
    
    if number.lower() == '종료':
        print("마무리 하겠습니다")
        break
    
    try:
        n = int(number) % 10  # 입력 받은 숫자를 정수로 변환하고 마지막 자리를 가져옴

        if n == 0 or n == 2 or n == 4 or n == 6 or n == 8:
            print("짝수입니다")
        else:
            print("홀수입니다")

    except ValueError:
        print("숫자나 '종료'를 입력하세요.")
        break
    '''
    
while True:
    number = input("숫자를 입력하세요>")
   
    
    if number.lower() == '종료':
        print("마무리 하겠습니다")
        break
    
    try: 
        n = int(number)
        num = n[-1]               
        if n ==0 or n ==2 or n ==4 or n==6 or n==8:
            print("짝수입니다")
        else:
            print("홀수입니다")
    except ValueError:
            print("문자는 사용 금지입니다.")
            break 

# 짝수와 홀수 구분하기 
