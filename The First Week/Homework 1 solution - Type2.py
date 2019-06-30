''' Type2 1번 '''

a = [1, 2, 3, 4, 5, 11, 24, 1231, 12412312, 1231113, 222221, 0, -1, 2, 3, 1]

### 1-1)
print(sum(a))

### 1-2)
#원자료를 보존하기 위해 카피 
a_copy = a.copy()
#오름차순
a_copy.sort()
print(a)
#내림차순
a_copy.sort(reverse = True)
print(a)

### 1-3)
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



### 1-4)
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


''' Type2 2번 '''

#### 데이터 만들기 ####
week_day = ['FRI', 'SAT', 'SUN', 'MON', 'TUE', 'WED', 'THU']
week_day = week_day * 53
week_day = week_day[0:365]
year = 2016
month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
day_type_1 = list(range(1, 32))
day_type_2 = list(range(1, 31))
day_type_3 = list(range(1, 30))
sequence = list(range(1, 366))

#### 날짜 조합하기 ####
all_day = []
for i, month_content in enumerate(month):
    if month_content in [1, 3, 5, 7, 8, 10, 12] :
        for day_t_1 in day_type_1:
            all_day.append('%d-%d-%d' %(year, month_content, day_t_1))
    elif month_content in [4, 6, 9, 11] :
        for day_t_2 in day_type_2:
            all_day.append('%d-%d-%d' %(year, month_content, day_t_2))
    elif month_content  == 2 :
        for day_t_3 in day_type_3:
            all_day.append('%d-%d-%d' %(year, month_content, day_t_3))

import numpy as np
import pandas as pd

#### 데이터 준비 ####
all_day = np.array(all_day)
sequence = np.array(sequence)
week_day = np.array(week_day)



#### 함수 제작 ####
# 해당 범위에 없는 month 나 day가 들어가게 되면 에러가 뜸
def date(month, day):
    try :
        for i, month_day in enumerate(all_day):
            if month_day == '2016-%s-%s' %(month, day):
                seq = i
                return week_day[i]
            else:
                pass
        return print('There is no matched date in dataframe') 
    except:
        pass





