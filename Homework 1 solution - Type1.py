''' Type1 1번 '''

## solution1-1
sum_all = 0
for i in range(1001):
    sum_all += i
print(sum_all)

## solution1-2
average_val = (1000 + 1)/2
sum_all = average_val * 1000


''' Type1 2번 '''

import numpy as np
print(np.log(sum_all))


''' Type1 3번 '''

a = [1, 2, 3, 4, 5, 11, 24, 1231, 12412312, 1231113, 222221, 0, -1, 2, 3, 1]

### 3-1)
print(sum(a))

### 3-2)
#원자료를 보존하기 위해 카피 
a_copy = a.copy()
#오름차순
a_copy.sort()
print(a)
#내림차순
a_copy.sort(reverse = True)
print(a)

### 3-3)
for i, content in enumerate(a):
    # i는 a content의 순서
    print(i)
    # content는 a 안에 있는 요소의 내용
    print(content)

# 홀수의 개수는?
count_odds = 0
for i, content in enumerate(a):
    # 요소를 나누기 2 했을 때 나머지가 0이 아니고 0보다 큰수 
    if content % 2 != 0 and content >= 0 :
       count_odds += 1

# 짝수의 개수는?
count_evens = 0
for i, content in enumerate(a):
    # 요소를 나누기 2 했을 때 나머지가 0이고 0보다 큰수
    if content % 2 == 0 and content >= 0 :
        count_evens +=1



### 3-4)
#sol1) 1000보다 큰수
count_more_than_1000 = 0
for i in range(len(a)):
    if a[i] >= 1000 :
        count_more_than_1000 += 1
    else:
        pass
print(count_more_than_1000)

#sol2) 1000을 요소에서 뺐을 때 양수인 숫자
count_more_than_1000 = 0
for i in range(len(a)):
    if a[i] - 1000 >= 0 :
        count_more_than_1000 += 1
print(count_more_than_1000)

#sol3) 숫자의 개수
count_more_than_1000 = 0
for i in range(len(a)):
    if len(str(a[i])) >= 4 :
        count_more_than_1000 += 1
    else:
        pass
print(count_more_than_1000)





