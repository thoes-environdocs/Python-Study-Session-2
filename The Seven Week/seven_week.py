## 1. 2016년도 .csv파일 합치기

import os
os.chdir("C:\\Users\\admin\\Desktop\\실습\\dust_2016")


import glob
file_names = glob.glob("*.csv")


import pandas as pd
data_2016 = []
for i, content in enumerate(file_names):
    data_2016.append(pd.read_csv(content, encoding = "CP949"))


data_2016 = pd.concat(data_2016, ignore_index = True)


## 2. 데이터의 길이, 컬럼의 속성을 알아보고
## unique한 'region, code, 주소'뭐가 있는지 알아보시오

length_2016 = len(data_2016)
column_list = data_2016.columns.tolist()


import numpy as np
region_list = np.unique(data_2016['region'])
code_list = np.unique(data_2016['code'])
address_list = np.unique(data_2016['주소'])


## 3. 'date'컬럼의 '년도', '월', '일'의 컬럼을 만드려면 어떻게 해야할까?

# 데이터 만들어 주기
year = [int(str(data_2016['date'][i])[0:4]) for i in range(len(data_2016))]
year = np.array(year)

month = [int(str(data_2016['date'][i])[4:6]) for i in range(len(data_2016))]
month = np.array(month)

# 데이터 붙이기
data_2016['year'] = year
data_2016['mm'] = month

# 컬럼 순서 바꾸기
data_2016 = data_2016[['region', 'code', 'site', 'date','year', 'mm', 'SO2', 'CO', 'O3', 'NO2', 'dust', 'PM25', '주소']]


## 4. 각 장소별 2016년도 월별 평균 만들기

#1) 장소 한개로 생각해 보기
##region_1_data = data_2016[data_2016['region'] == region_list[0]]
##
##region_1_month = np.unique(region_1_data['mm'])
##
##
##region = region_list[0]
##year = 2016
##
##
##month_real = []
##
##for i, content in enumerate(region_1_month):
##    mean_data = [region, year, content]
##    mean_month = region_1_data[region_1_data['mm'] == region_1_month[i]]
##    mean_number = np.mean(mean_month.iloc[:, 6:12])
##
##    
##    mean_data.extend(mean_number)
##
##    month_real.append(mean_data)
##
##
##month_real = pd.DataFrame(month_real)
##month_real.columns = ['region', 'year', 'mm', 'SO2', 'CO', 'O3', 'NO2', 'dust', 'PM25']
##    
##    

##4. 각 장소별 2016년도 월별 평균 만들기

#2) 장소 한꺼번에 생각하기

month_real = []

for j, region_content in enumerate(region_list):
    region_data = data_2016[data_2016['region'] == region_list[j]]
    region_month = np.unique(region_data['mm'])

    year = 2016

    for i, month_content in enumerate(region_month):
        
        mean_data = [region_content, year, month_content]
        mean_month = region_data[region_data['mm'] == region_month[i]]
        mean_number = np.mean(mean_month.iloc[:, 6:12])

        mean_data.extend(mean_number)

        month_real.append(mean_data)
        
month_real = pd.DataFrame(month_real)
month_real.columns = ['region', 'year', 'mm', 'SO2', 'CO', 'O3', 'NO2', 'dust', 'PM25']
    



##5-1 위에서 만든 데이터 프레임을 통해 dust와 PM25의 일대일 플랏을 그려 보시오
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows = 1, ncols = 1)

ax.scatter(month_real['dust'], month_real['PM25'], color ='red')
ax.set_xlabel("dust")
ax.set_ylabel("PM25")
ax.set_title("dust~PM25")

plt.show()

##5-2 correlation plot
import seaborn as sns
month_real_for_corr = month_real.loc[:, 'SO2' : 'PM25']
corr = month_real_for_corr.corr(method = 'pearson')
print(corr)
ax = sns.heatmap(corr, annot = True, cmap ="YlGnBu")
plt.show()

##5-3 월별 박스 플랏
import seaborn as sns
sns.boxplot(x = "mm" , y = "dust", data = month_real)
sns.boxplot(x = "mm", y = "PM25", data = month_real)
plt.show()


## 6.
os.chdir("C:\\Users\\admin\\Desktop\\실습")
dir_names = glob.glob("*/")


data_all = []

for i, dir_content in enumerate(dir_names):
    os.chdir("C:\\Users\\admin\\Desktop\\실습\\%s" %(dir_content))
    
    file_names = glob.glob("*.csv")
    
    for j, file_content in enumerate(file_names):

        print(file_content)

        data = pd.read_csv(file_content, encoding = "CP949")

        data_all.append(data)

data_all = pd.concat(data_all, ignore_index = True)

        





