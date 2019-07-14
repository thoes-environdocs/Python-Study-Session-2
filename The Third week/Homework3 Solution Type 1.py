#1

def summation(a, b):
    value = sum(list(range(a, b+1)))
    return value


#2

def forwarding(list_data):
    if len(list_data) == 0 :
        return("The length of list is 0")
    elif len(list_data) == 1:
        return list_data
    elif len(list_data) > 1:
        new_list = [list_data[-1]] + list_data[:-1]
        return new_list


#3.1
## 리스트에서 요소가 1인 것의 개수
def counting(number, list_data):
    count = 0
    for i in range(len(list_data)):
        if list_data[i] == number:
            count +=1
    return count

# 3.2
def counting_element(what_number ,list_data):
    if what_number not in list_data:
        return print("no number in list")
    
    elif what_number in list_data:
        count_ = []
        import numpy as np
        number_list = list(set(list_data))
        number_list.sort()

        for i in range(len(number_list)):
            count_.append(counting(number_list[i], list_data))

        ## making tuple
        tuple_list = []
        for i in range(len(count_)):
            tuple_list.append((number_list[i], count_[i]))

        for j, content in enumerate(tuple_list):
            if content[0] == what_number:
                return tuple_list[j]


## develop for 4
##list_data =[6, 10, 5]
##list_data.sort()
##length = len(list_data)
##string_data = [ str(i) for i in list_data ]
##
##### 서로 바꾸었을 때 큰 것의 개수 
##count_all = []
##for i in range(length):
##    count = 0
##    for j in range(length):
##        if int(string_data[i] + string_data[j]) > int(string_data[j] + string_data[i]):
##            count += 1
##
##    count_all.append(count)
##
##### count_all을
##tuple_data = []
##for i in range(length):
##    t_data = (string_data[i], count_all[i])
##    tuple_data.append(t_data)
##
##tuple_data.sort(key = lambda x : x[1], reverse = True)
##
##k = ''
##for i in range(length):
##    k += tuple_data[i][0]

# 4. 
def solution(list_data):
    list_data.sort()
    length = len(list_data)
    string_data = [ str(i) for i in list_data ]

    ### 서로 바꾸었을 때 큰 것의 개수 
    count_all = []
    for i in range(length):
        count = 0
        for j in range(length):
            if int(string_data[i] + string_data[j]) > int(string_data[j] + string_data[i]):
                count += 1
        count_all.append(count)

    ### count_all을
    tuple_data = []
    for i in range(length):
        t_data = (string_data[i], count_all[i])
        tuple_data.append(t_data)

    tuple_data.sort(key = lambda x : x[1], reverse = True)

    k = ''
    for i in range(length):
        k += tuple_data[i][0]


    return k


##if __name__ == '__main__':
##    list_data =[3, 30, 34, 5, 9]
##    print(solution(list_data))

