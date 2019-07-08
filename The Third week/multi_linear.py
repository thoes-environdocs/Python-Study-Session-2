
################################################
###############Linear Regression################
################################################


''' 머신러닝을 통한 다중 회귀 '''
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np



### 데이터 생성 ### 
import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\Desktop\\waterquality.csv", encoding = "CP949")

new_data = data[(data['yy']>2000) &
                (data['t.coli'] > 0) &
                (data['do'] > 0) &
                (data['f.coli'] > 0)]

print(new_data.head())
print(len(new_data))

t_coli = np.log(new_data['t.coli'])
do = np.log(new_data['do'])
f_coli = np.log(new_data['f.coli'])


x = np.array([do, f_coli])
x = x.T
y = np.array(t_coli)

## StandardScaler()로 평균이 0, 분산이 1로 데이터 분포도 변환
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = np.ravel(scaler.fit_transform(y.reshape(len(y), 1)))


### Training, Test ###
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)

### Optimization Method : Regression ###
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 1000, learning_rate = "constant",
                       eta0 = 0.0001, verbose = 1, penalty = None,
                       n_iter = 100, warm_start = True)

## learning_rate = 'optimal' ##

#sgd_reg.fit(x_train, y_train)

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

# 상수항 #
print(sgd_reg.coef_)
# 회귀 계수 #
print(sgd_reg.intercept_)

X_1 = np.linspace(min(x.T[0]), max(x.T[0]))
X_2 = np.linspace(min(x.T[1]), max(x.T[1]))
X_1, X_2 = np.meshgrid(X_1, X_2)
Y = sgd_reg.coef_[0] * X_1 + sgd_reg.coef_[1] * X_2 + sgd_reg.intercept_

fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.plot_surface(X_1, X_2, Y, alpha = 0.5, cmap = 'binary')
ax.plot(x_train.T[0], x_train.T[1], y_train, 'r.')
ax.plot(x_test.T[0], x_test.T[1], y_test, 'g.')

ax.view_init(elev = 31, azim = -134)
ax.set_ylim(min(x.T[0]), max(x.T[1]))
ax.set_xlim(min(x.T[0]), max(x.T[0]))
ax.set_zlim(min(y), max(y))
ax.set_xlabel("X_1 , do")
ax.set_ylabel("X_2, f_coli")
ax.set_zlabel("Y, t_coli")
plt.show()


### 성능 평가 ###
def RMSE(regression_y, real_y):
    n=len(real_y)
    sum_val = 0
    for i in range(n):
        diff_square = (real_y[i][0] - regression_y[i][0])**2
        sum_val += diff_square
    rmse = (sum_val/n)**0.5
    return rmse

regression_y = sgd_reg.coef_[0] * x_test[:, 0] + sgd_reg.coef_[1] * x_test[:, 1] + sgd_reg.intercept_
regression_y = regression_y.reshape(len(regression_y), 1)
real_y = y_test.reshape(len(y_test), 1)

RMSE_val = RMSE(regression_y, real_y)
print(RMSE_val)

plt.plot(real_y, regression_y, 'go')
X = np.linspace(-5, 5, 5)


plt.plot(X, X)
plt.xlabel("real_y")
plt.ylabel("regression_y")
plt.show()

