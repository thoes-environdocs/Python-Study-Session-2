### sol 1
a = list(range(1, 1001))

### sol 2
a = list(range(1, 1001))
b = list(range(2, max(a)+1, 2))

a = list(set(a) - set(b))

### sol 3
a = list(range(1, 1001))
b = list(range(3, max(a) + 1, 3))

a = list(set(a) - set(b))

### sol 4 소수판별기
def prime_number(number):
    if number !=1:
        count = 0
        for f in range(2,number):
            if number % f == 0:
                count += 1
            if count == 1:
                return False
    else:
        return False 
    return True

### sol 5 소수 배수 제거
# 시간체크를 위한 모듈
import time

def removal_prime(k):
    start = time.time()
    
    a=[]
    for i in range(2, k+1):
        if prime_number(i):
            a.append(i)

    print("time :", time.time() - start)
    
    return len(a)


'''BUT 시간이 너무 오래걸리는 알고리즘임 '''

''' 어떻게 시간을 단축할지 생각을 해봐야함 ''' 


### sol 6, 7

### 시간을 제는 모듈
import time

### 알고리즘 k 숫자에 대한 소수 구하는 알고리즘

def algorithm_sosu(k):
    start = time.time()
    
    a = list(range(2, k+1))
    j = 0
    while True :
        
        
        if j >= len(a):
            break

        
        max_a = max(a)
        num = a[j]
        if True:
            ### 배수리스트 생성
            b = list(range(num*2, max_a+1, num))

            ### a에서 배수리스트 제거
            if len(b) != 0 :
                candidate = list(set(a) - set(b))
                
                if len(a) != len(candidate):
                    a = candidate
                    a.sort()

        
        j = j + 1

    print("time :", time.time() - start)
    
    return len(a)



