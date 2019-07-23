# 1. 정수 -> 실수 (오타)
import numpy as np
list_data = np.linspace(0, 25, 21)

#2. 반올림하기
list_data = np.round(list_data)

#3. 어떤 실수 두 숫자 사이
def btw_two(num, list_data):
    for i, content in enumerate(list_data):
        if i != len(list_data):
            if list_data[i] <= num < list_data[i+1]:
                front = np.round(list_data[i])
                back = np.round(list_data[i+1])
        else: pass
    return front, back

#4. 큰것을 Max(1) 작은 것을 Min(0)
num = 8.1
front, back = btw_two(num, list_data)
def MinMaxScaler(num, min_val, max_val):
    scale = (num - min_val)/(max_val - min_val)
    return scale

print(MinMaxScaler(num, front, back))

#5. 어디로 가야하나?
def which_num(num, list_data):
    front, back = btw_two(num, list_data) ## 2의 함수
    if np.round(MinMaxScaler(num, front, back)) == 0: ## 3의 함수
        return front
    else:
        return back

#6. 8.1은 어디로 가야하나?
print(which_num(8.1, list_data))

#7. a, b 숫자 범위, n개의 정수간격, num의 실수 어디로 변환?
def exchange_num(a, b, n, num):
    list_data = np.linspace(a, b, n)
    answer = which_num(num, list_data) #5의 함수 
    return answer

#8. 색깔을 매치 시켜 보시오 a<num<b
color=["#ff0000","#ff0803","#ff0f05","#ff1708","#ff1f0a","#ff260d",
       "#ff2e0f","#ff3612","#ff3d14","#ff4517","#ff4c1a","#ff541c",
       "#ff5c1f","#ff6321","#ff6b24","#ff7326","#ff7a29","#ff822b",
       "#ff8a2e","#ff9130","#ff9933"]

def which_color(a, b, n, num, color):
    return color[int(exchange_num(a, b, n, num))]
    

#9. nan채우기
list_data = [1, 'nan', 2, 'nan', 'nan', 7, 9, 'nan', 12, 'nan', 'nan', 20]

num_index = []
string_index = []
for i in range(len(list_data)):
    if type(list_data[i]) == str:
        string_index.append(i)
    else:
        num_index.append(i)

for i in range(len(num_index)):
    for j in range(len(string_index)):
        if num_index[i] < string_index[j] < num_index[i+1]:
            # 등차 수열의 합
            list_data[string_index[j]] = list_data[num_index[i]] + (list_data[num_index[i+1]] - list_data[num_index[i]])/(num_index[i+1] - num_index[i]) * (string_index[j] - num_index[i])

# 그냥 넣은거 
print(list_data)
# 반올림 해서 넣은것
print(np.round(list_data, 2))
