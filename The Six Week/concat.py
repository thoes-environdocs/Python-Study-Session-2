import pandas as pd
import numpy as np

## 1. 단순 병합하기
'''
pd.Series는 pandas에서 데이터의 리스트 개념임

numpy 형식의 데이터를 데이터 프레임에 넣을 땐,
pd.Series로 변환 후 형식을 맞추어서 넣어주어야 함.

'''

s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])

s3 = np.array([1, 23])
s3 = pd.Series(s3)

### 인덱스 유지 하면서 붙이기 
data_list_1 = pd.concat([s1, s2, s3])
### 인덱스 구분 없이 붙이기 
data_list_2 = pd.concat([s1, s2, s3], ignore_index = True)
### 인덱스, key 구분 하여 붙이기
data_list_3 = pd.concat([s1, s2, s3], keys=['s1', 's2', 's3'])
### 인덱스, key 구분, 이름 붙이기
data_list_4 = pd.concat([s1, s2, s3], keys = ['s1', 's2', 's3'],
                        names = ['Series', 'RowID'])

data_list_4.keys()

## 2. join "outer", "inner"
'''
outer은 합집합을 의미한다.
inner은 교집합을 의미한다.
'''

df1 = pd.DataFrame([['a', 1], ['b', 2]],
                    columns=['letter', 'number'])
df2 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
                    columns=['letter', 'number', 'animal'])

concat_outer = pd.concat([df1, df2])
concat_inner = pd.concat([df1, df2], join = "inner")
