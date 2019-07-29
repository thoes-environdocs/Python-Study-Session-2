######## 데이터 불러오기 ########

'''
1. 환경 변수 설정

'''

import os

## 현재의 경로는 어디에 있지?
os.getcwd()


## 현재의 경로를 바꾸고 싶어
os.chdir("C:\\Users\\admin\\Desktop\\파이썬 스터디 세션2\\6주차\\example")



'''
Q1 :  practice로 경로를 바꾸어 보시오

'''


'''
2. Data Genegrating - 엑셀 파일 불러오기

주의할 점! 데이터의 Column명을 동일하게 맞추어 데이터를 구성하는 것이
           데이터를 합치거나 원하는데로 가공할 때 편하다.

'''

## 기본 pandas
import pandas as pd
data = pd.read_csv("Forest.csv", encoding = "CP949")
data = pd.read_excel("Agriculture.xls", encoding = "utf-8")

#상위 5개의 데이터를 보여줌 
data.head()
#데이터를 summary해줌 mean, std, 25%, 50%, 75%, max
data.describe()

'''
Q2 : practice에서 Urban.xls 데이터를 불러와 보시오
Q3 : 위 데이터를 요약해 보시오

'''

## 심화 pandas with glob
import glob
'''
glob는 파일들의 목록을 뽑을 때 사용하게 된다.
목록을 뽑을 수있다면 한꺼번에 데이터를 불러 올수 있어 매우 편리하다.
사람이 클릭을 여러번 해야할 일을 한꺼번에 실행해주어 시간을 절약할 수 있다.

'''
file_names = glob.glob("*")
file_names = glob.glob("*.csv")
file_names = glob.glob("*.xlsx")
file_names = glob.glob("*.xls")


'''
Q4 : pandas와 glob를 이용하여 경로의 csv파일을 모두 불러와 리스트에 넣어보시오.
Q5 : pandas와 glob를 이용하여 경로의 xls파일을 모두 불러와 리스트에 넣어보시오.

'''
#### answer Q4, Q5
file_names = glob.glob("*.csv")
data_list = []
for i in range(len(file_names)):
    data_csv = pd.read_csv(file_names[i], encoding = "CP949")
    data_list.append(data_csv)

'''
Q6 : data_list에 있는 각각의 파일 마다의 길이를 프린트 해보시오.

'''

#### answer Q6
for i in range(len(data_list)):
    print( '%d번째의 데이터 길이는 %d입니다.' %(i, len(data_list[i])))


### data_list에 있는 모든 데이터 합치기
# 아래로 붙이기 
data_all_down = pd.concat(data_list, ignore_index= True)
# 옆으로 붙이기
data_all_right = pd.concat(data_list, axis = 1, ignore_index = True)

'''
Q7 : xls형식의 파일을 불러와서 합쳐진 데이터 프레임을 만들어 보시오

'''

'''
3. 데이터 Columns와 Rows
Columns 데이터가 가지고 있는 특징 (= 필드, 열, 속)
Rows 데이터 구분을 해줌 (= 레코드, 행, index)
데이터는 행과 열에 의해 표현이 가능하다! 
'''
import numpy as np
index_list = data_all_down.index.tolist()
index_list = np.array(index_list)
columns_list = data_all_down.columns.tolist()
columns_list = np.array(columns_list)

'''
Q8 : 데이터를 옆으로 붙인 것의 index와 columns의 list를 확인해 보시오
'''

'''
4. Columns 및 Row의 정보를 통해 데이터의 추출

### 데이터엔 서열이 존재한다 !
### 데이터를 구성하는 사람이 되어서 생각해 보자
### 수질 데이터를 만들 때, 어떤 데이터를 가장 먼저 제1의 속성으로 만들어야 할까?
### 무엇이 가장 큰 범주 일까?
### 장소 -> 일자 -> 수질 항목
'''

# 4.1 기본 column 이름으로 추출
data_all = pd.concat(data_list, ignore_index= True)


print(data_all.columns.tolist())

## 참고: column 위치 바꾸기
data_all = data_all[['site', 'region', 'date', 'ph', 'do', 'bod', 'cod',
                     'ss', 'tn', 'tp', 'temp', 'ec', 't.coli', 'nh3', 'no3',
                     'po4', 'f.coli', 'landuse']]

data_all = data_all[['region', 'site', 'date', 'ph', 'do', 'bod', 'cod',
                     'ss', 'tn', 'tp', 'temp', 'ec', 't.coli', 'nh3', 'no3',
                     'po4', 'f.coli', 'landuse']]

extract_column_1 = data_all[['tp']]
extract_column_2 = data_all[['cod']]
extract_columns = data_all[['tp', 'cod']]

'''
Q9 : 데이터의 t.coli, f.coli, no3의 정보를 뽑아보자

'''
extract_columns = data_all[['t.coli', 'f.coli']]


# 4.2 심화 iloc, loc
## iloc 숫자 기반 loc 정보 기반 

## 4.2.1 loc

## 정보 기반 loc[행, 열] 리스트를 안에다가 넣어 주면 된다.

### 열 정보를 전체 다 가져올래
data_all.loc[[1], :]

### 1번째 row의 정보와 'tp'라는 컬럼을 가지고 있는 데이터를 가져오자
data_all.loc[[1], ['tp']]

### 1번째, 2번째 row의 정보와 'tp'라는 컬럼을 가지고 있는 데이터를 가져오자
data_all.loc[[1, 2], ['tp']]

### 1번째, 2번째 row의 정보와 'tp', 'cod'라는 컬럼을 가지고 있는 데이터를 가져오자
data_all.loc[list(range(1,3)), ['tp', 'cod']]

'''
Q10 :  데이터의 row가 10, 12, 13, 212이고 column이 't.coli, cod, bod'인 데이터를 추출하시오
Q11 :  row가 홀수인 데이터만 다 가져오고 싶어
'''
data_all.loc[[10, 12, 13, 12000], ['t.coli', 'cod', 'bod']]
data_all.loc[list(range(1, len(data_all), 2)), :]


## 4.2.2 iloc

## 숫자 기반 iloc[행, 열]리스트를 안에다가 넣어 주면 된다. 
data_all.iloc[1, 1]
data_all.iloc[:, 1] # data_all['site']와 기능이 같다. 

data_all.iloc[[1,3,4,2,4,1], [1,2]]

'''
Q12 : 데이터의 row가 짝수이고, column이 홀수인 데이터만 다가져오고 싶어
'''
data_all.iloc[list(range(1, len(data_all), 2)), list(range(1, len(data_all.columns.tolist()), 2))]

'''
5 조건을 통해 데이터 추출
원하는 조건을 통해 데이터를 추출해본다. 
'''
## 참고 : 매우 중요 
region = np.unique(data_all['region'])

'''
Q13 : 데이터의 site는 어떤 것이 있을까? 리스트로 구성해 보시오

'''
### answer 13
site = np.unique(data_all['site'])


## 원리 조건에 맞는 데이터를 True, False로 인식함 True인 데이터만 뽑아온다.
data_all['region'] == '광주_황룡강합류전' ## true인것을 만들기

data_all[data_all['region'] == '광주_황룡강합류전'] ## true 뽑아오기


data_all[data_all['region'].isin(['고양_공릉천상류', '고양_창릉천', '광주_광주천'])]
data_all[data_all['landuse'].isin(['Agriculture', 'Urban'])]


#교집합
data_all[(data_all['landuse'].isin(['Agriculture'])) & (data_all['t.coli'] > 1000)]

#합집합
data_all[(data_all['landuse'].isin(['Agriculture'])) | (data_all['t.coli'] > 1000)]


## 참고 : 날짜 변환
date_split = [data_all['date'][i].split('-') for i in range(len(data_all))]
date_split = pd.DataFrame(date_split)
data_all['year'] = date_split.iloc[:,0]
data_all['mm'] = date_split.iloc[:,1]
data_all['dd'] = date_split.iloc[:,2]
print(data_all.columns.tolist())
data_all = data_all[['region', 'site', 'date','year', 'mm', 'dd', 'ph', 'do', 'bod', 'cod',
                     'ss', 'tn', 'tp', 'temp', 'ec', 't.coli', 'nh3', 'no3',
                     'po4', 'f.coli', 'landuse']]


'''
Q14 : data_all에서 year가 2018인 데이터를 추출하여 data_2018인 객체를 만들어 보시오
Q15 : data_all에서 year가 2014, 2018이고 site가 '갑천3', '갑천4', '갑천5'인 데이터를 뽑으시오
Q16 : region을 '_'를 통해 split한다음 앞의 것을 'region_1', 뒤의 것을 'region_2'로 만들어 data_all 데이터 프레임에 추가하시오
Q16 : data_all에서 t.coli가 1000보다 크고, f.coli가 1000보다 크고, region_1이 '서울' 인 데이터를 뽑으시오

'''
region_split = [data_all['region'][i].split('_') for i in range(len(data_all))]
region_split = pd.DataFrame(region_split)
data_all['region_1'] = region_split.iloc[:,0]
data_all['region_2'] = region_split.iloc[:,1]
data_all = data_all[['region','region_1', 'region_2', 'site', 'date','year', 'mm', 'dd', 'ph', 'do', 'bod', 'cod',
                     'ss', 'tn', 'tp', 'temp', 'ec', 't.coli', 'nh3', 'no3',
                     'po4', 'f.coli', 'landuse']]

'''
6. 데이터를 Sorting해본다.
'''
## 사이트 명으로 오름차순
data_all = data_all.sort_values(by = ['site'])
## 사이트 명으로 내림차순 
data_all = data_all.sort_values(by = ['site'], ascending = False)
## site -> year -> mm 순서대로
data_all = data_all.sort_values(by = ['site', 'year', 'mm'])


'''
7. 데이터 프레임을 제거한다.
수정하기 전엔 원본 데이터는 보존 해야 한다.
따라서 카피본을 먼저 만들고 그 데이터를 수정하도록 한다.
'''

data_all_copy = data_all.copy()
## 컬럼 제거 
data_all_copy = data_all_copy.drop(['tp'], axis = 1)
## 행 제거
data_all_copy = data_all_copy.drop([1, 2, 3], axis = 0)

print(len(data_all_copy))
data_all_copy = data_all_copy.drop(data_all_copy[data_all_copy['site'].isin(['갑천3'])].index.values)


'''
Q17 : data_all에서 대전 지역의 데이터를 drop시켜 보시오

'''

region_1 = np.unique(data_all['region_1'])
print(len(data_all_copy))
data_all_copy = data_all_copy.drop(data_all_copy[data_all_copy['region_1'].isin(['대전', '서울'])].index.values)
print(len(data_all_copy))


'''
8. 데이터 프레임을 추가 해봅시다.

.append로 추가를 해주어 보자
.concat도 사용가능함

'''
k = data_all_copy.iloc[1,:]
data_all_copy = data_all_copy.append(k, ignore_index = True)
type(k)
type(data_all_copy)

k = pd.DataFrame.transpose(pd.DataFrame(k))
data_all_copy = pd.concat([data_all_copy, k], ignore_index = True)


'''
9. 저장하기
'''
writer = pd.ExcelWriter('example_1.xlsx')
data_all_copy.to_excel(writer,'Sheet1')
# data_all_copy.to_csv(writer, 'Sheet1')
writer.save()
