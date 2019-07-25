# Sanghun
def change(price):
    coin=[500,100,50,10]
    money = 1000 - price
    coinNumber = 0

    for i in coin:
        coinNumber = coinNumber + (money // i)
        money = money - (money // i)*i
    return coinNumber

# Theos
def cost(value):
    #500원
    count_500 = 0
    for i in range(0, int(value/500)):
        if value >= 500 :
            value  = value - 500
            count_500 += 1
    #100원
    count_100 = 0
    for i in range(0, int(value/100)):
        if value >= 100:
            value = value - 100
            count_100 += 1
    #50원  
    count_50 = 0
    for i in range(0, int(value/50)):
        if value >= 50 :
            value = value - 50
            count_50 += 1
    #10원   
    count_10 = 0
    for i in range(0, int(value/10)):
        if value >= 10:
            value = value - 10
            count_10 += 1
        
    return count_500, count_100, count_50, count_10
        
# Theos
import math
num = '11010101010100'

def split(num):
    string = str(num)
    split_data = []
    for i in range(len(string)):
        if i == len(string)-1: pass
        else:
            if string[i] != string[i+1]:
                split_data.append(i)
    return split_data
                
split_data = split(num)

def split_list(num, split_data):
    split_split = []
    for i, content in enumerate(split_data):
        if i == 0:
            split_split.append(num[0:content+1])
        elif i != 0:
            if i == len(split_data)-1 :
                split_split.append(num[ split_data[i-1]+1 : split_data[i]+1 ])
                split_split.append(num[split_data[i]+1:])
            else:
                split_split.append(num[split_data[i-1]+1 : split_data[i]+1])
    return split_split

print(math.ceil(int((len(split_list(num, split_data))-1) / 2))))

#재혁  중요! *****
import math

turnlist = input ("series of 7 numbers")
splited = list()
for char in turnlist:
    splited.append(int(char))
print(splited)

differ = list()
for i, x in enumerate(splited) :
    if i > 0 :
        if splited[i] != splited[i-1]:
            differ.append(1)
        else:
            differ.append(0)
print(differ)

turn = 0
for x in differ:
    turn += x
print( math.ceil(turn/2))
