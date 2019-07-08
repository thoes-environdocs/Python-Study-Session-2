################################################
###############Linear Regression################
################################################


''' 머신러닝을 통한 선형회귀 '''

import numpy as np
from sklearn.linear_model import SGDRegressor


### Data Generating ### 
x = 2* np.random.rand(100, 1)
y = 4+5*x + np.random.randn(100, 1)
y = np.ravel(y)

### training, test #### 7:3 또는 8:2로 나눔
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)


### Optimization Method : Regression ###
sgd_reg = SGDRegressor(max_iter = 10, learning_rate = "constant",
                       eta0 = 0.001, verbose = 2, penalty = None,
                       n_iter = 100, warm_start = True)

'''
-------------SGDRegressor------------
max_iter : 최대로 훈련을 시키는 횟수
learning_rate : 학습률
eta0: 학습률을 결정하는 요소
verbose : 반복학습이 진행될때 필요한 정보를 보여주게 됨
penalty : l1규제 l2규제, Lasso, Ridge를 할 순 있긴 한데 다른 Regressor 존재함
          기본적으로 None을 꼭 해주어야 규제가 들어가지 않은 학습을 진행할수 있다.
n_iter_no_change : 목적함수의 값이 원하는 갯수 만큼 변하지 않는다면 그 모델을 성립된 모델로 간주함
warm_start : 훈련시킨 회귀분석을 지속해서 훈련을 시키고 싶을 때 사용함.
--------------------------------------
'''
### fitting ###
#sgd_reg.fit(x, y)


### loss 함수의 history 추출 ### 
import sys
from io import StringIO
string_list = sys.stdout
sys.stdout = mystdout = StringIO()

### fitting ###
sgd_reg.fit(x_train, y_train)

sys.stdout = string_list
loss_history = mystdout.getvalue()

loss_list = []
sequence = []
for seq, line in enumerate(loss_history.split("\n")):
    # loss를 한 5씩 끊어서 보고 싶음
    if (seq + 1) % 5 == 0:
        
        if(len(line.split("loss: ")) == 1):
            continue
        
        sequence.append(((seq+1) +2)/3+1)
        loss_list.append(float(line.split("loss: ")[-1]))


### plotting ###
import matplotlib.pyplot as plt
plt.plot(sequence, loss_list, 'g-')
plt.plot(sequence, loss_list, 'ro')
plt.show()
###################################################

# 상수항 #
print(sgd_reg.coef_)
# 회귀 계수 #
print(sgd_reg.intercept_)

X = np.linspace(0, 2, 10)
Y = sgd_reg.coef_ * X + sgd_reg.intercept_

### training된것 만 ### 
plt.plot(X, Y, 'b-')
plt.plot(x_train, y_train, 'ro')
plt.show()

### test된것도 포함 ### 
plt.plot(X, Y, 'b-')
plt.plot(x_train, y_train, 'ro')
plt.plot(x_test, y_test, 'go')
plt.show()

### 성능 평가 ###
def MSE(regression_y, real_y):
    n=len(real_y)
    sum_val = 0
    for i in range(n):
        diff_square = (real_y[i][0] - regression_y[i][0])**2
        sum_val += diff_square
    mse = (sum_val/n)
    return mse

regression_y = sgd_reg.coef_ * np.array(x_test) + sgd_reg.intercept_
real_y = y_test.reshape(len(y_test), 1)

MSE_val = MSE(regression_y, real_y)
print(MSE_val)
    
plt.plot(real_y, regression_y, 'go')
X = np.linspace(0, 15, 5)

plt.xlim(0, 15)
plt.ylim(0, 15)
plt.plot(X, X)
plt.xlabel("real_y")
plt.ylabel("regression_y")
plt.show()




