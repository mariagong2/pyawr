# -*- coding: utf-8 -*-
"""
프로그래밍의 오류의 종류 두가지 
1. 프로그램 실행전에 발생하는 오류 -> 구문오류
2. 프로그램 실행후에 발생하는 오류 -> 런타임 오류 = 예외
2번의 경우 복잡하게 해결해야할때가 많은데, 이것을 해결하는것을
예외처리! 
예외처리 방법은 
1. 조건문을사용하는 법 -> 기본예외처리
2. try 구문을 사용하는법


#구문 오류의 예시 
print("프로그램이 실행됐다")
list_a[0] -> 이 경우 list_a에 대해서 선언되지않았기 때문에 에러가 발생 

#조건문으로 예외를 처리하는 방법에 대한 예시

number_input_a = int(input("정수입력해주세요>"))
print("원의반지름",number_input_a)
print("원의둘레", 2*3.14*number_input_a)
print("원의 넓이", 3.14*number_input_a*number_input_a)
#만약 정수로 입력하지 않는다면 에러발생 > ValueError: invalid literal for int() with base 10: '3센티'

'''
"""
'''
# 이러한 예외 에러를 처리하기위해서 조건문을사용하여 보자 
while True: # 무한루프 
    
    user_input_a = input("정수입력>")
    
    if user_input_a.isdigit():# 만약 숫자로만 입력했다면
        number_input_a = int(user_input_a)
        print("원의반지름",number_input_a)
        print("원의둘레", 2*3.14*number_input_a)
        print("원의 넓이", 3.14*number_input_a*number_input_a)

    else: 
        print("정수를 입력하지 않았습니다")
        break
'''  
'''
# try except 구문에대해서 파악해보기 
# 위와 같이 if문을 사용해도 되는데 try except 문은 에러가 있을지 없을지
#모르는상황에서 에러가 발생햇을때 동작시키는것을 except로 실행시킨다
while True:
    try:
        number_input_a=int(input("정수를 입력하세요"))
        print("원의반지름",number_input_a)
        print("원의둘레", 2*3.14*number_input_a)
        print("원의 넓이", 3.14*number_input_a*number_input_a)

    except: 
        print("잘못 입력했습니다 정수를 입력해주세요")

'''

# try except와 pass 문 조합하기 
# except 뒤에 pass를 써야한다 비워두면안되기 때문이다 
# 

 


