'''
pandas 종합 문제

Stage 1

경로를 Practice로 설정한다.

Stage 2

csv파일을 모두 불러와서 합친다.
가능하다면 합치면서 Region이라는 컬럼을 추가로 만들어 보아라 

Stage 3

전체 데이터의 개수 및 데이터 속
이 무엇이 있나 확인한다. (len columns 이름)

Stage 4

전체 데이터 중 'm/t'라는 컬럼 중 '지'인 데이터만을 추출하여 따로 데이터 프레임을 만든다.

Stage 5


각 지역마다 '지'라는 데이터는 각각 몇개가 있는가?
'''

### answer 18
#Stage 1
import os 
os.chdir("C:\\Users\\admin\\Desktop\\파이썬 스터디 세션2\\6주차\\practice")

#Stage 2
import glob
file_names = glob.glob("*.csv")

import pandas as pd
import numpy as np
data_list = []
for i in range(len(file_names)):
    file = pd.read_csv(file_names[i], encoding = "CP949")
    file['region'] = np.array([file_names[i].split('.')[0]] * len(file))
    data_list.append(file)

data = pd.concat(data_list, ignore_index = True)
data = data[['region', 'site', 'Y', 'X', 'm/t']]

#Stage 3
print(len(data))
print(data.columns.tolist())


#Stage 4
data_tributary = data[data['m/t'] == '지']


#Stage 5
region = np.unique(data['region'])
for i, content in enumerate(region):
    print(content, ':', len(data_tributary[data_tributary['region'] == content]))


#Stage 6
appending = ['busan', '태호태호', 35.193236999999996, 128.905055, '지']
appending = pd.DataFrame.transpose(pd.DataFrame(appending))
appending.columns = ['region', 'site', 'Y', 'X', 'm/t']
data_tributray = data_tributary.append(appending, ignore_index = True)

