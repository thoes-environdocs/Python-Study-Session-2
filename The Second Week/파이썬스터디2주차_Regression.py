
# coding: utf-8

# ### 데이터 준비

# In[10]:


import os
os.getcwd() #### 컴퓨터가 작업을 어디서 하는지 알려준다.


# In[11]:


import numpy as np ## 행렬을 쉽게 다루기 위한 모듈
import pandas as pd ## 엑셀을 쉽게 다루기 위한 모듈
import matplotlib.pyplot as plt ## 그림을 쉽게 다루기 위한 모듈 


# In[12]:


# 엑셀을 불러오고자 할때, data = pd.read_excel("~~", encoding = "cp949")
data = pd.read_csv("waterquality.csv", encoding = "cp949")


# In[13]:


data.head()


# In[14]:


data['ph'].head() #data중 Columnname이 ph인 것중 앞의 5개를 보여 줘라 
data['do'].head() #data중 Columnname이 do인 것중 앞의 5개를 보여 줘라 


# ### F.coli(분원성대장균)와 T.Coli(총 대장균)의 단순선형회귀 분석을 실시하고 싶다.

# In[15]:


# 2000년 이후의 T.Coli, F.Coli의 데이터 준비 데이터 중 2000년 이후, T.Coli가 0보다큰, F.Coli가 0보다 큰 데이터만을 사용하고 싶다.
new_data = data[(data['yy'] > 2000) & (data['t.coli'] > 0) & (data['f.coli'] > 0) ]
new_data.head()


# In[16]:


# 분원성 대장균을 통해 총 대장균의 정도를 예측가능한 모델을 만들고 싶다.
Y = np.log(new_data['t.coli'])
X = np.log(new_data['f.coli'])


# In[17]:


plt.plot(X, Y, 'o')
## 어느정도 상관성이 있다는걸 알게 되었음.


# ## 1.  정규 방정식을 이용한 선형 회귀 분석 

# #### 1.1 사이킷런을 이용하지 않은 분석

# In[18]:


### X, Y의 형식을 맞추어 줌
X = np.array(X).reshape(len(X), 1)
Y = np.array(Y).reshape(len(Y), 1)


# In[19]:


### X, Y가 어떻게 생겼는지 확인해야함. 
X[0:5]
Y[0:5]
print(len(X))
assert(len(X) == len(Y))


# In[20]:


## 1을 추가해 주는 이유는 상수항도 계수로 보기 위해서 W2*1 + W1*X = Y
X_b = np.c_[np.ones((5602, 1)), X]
X_b[0:5]


# ### W2*1 + W1*X = Y

# ![image.png](attachment:image.png) ##"정규 방정식: 수리적으로 계수를 구할 수 있다."##

# In[21]:


## 행렬의 곱을 해주는 연산
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
theta_best


# ### 4.13563846*1 + 0.75613179*X = Y (수학적으로 산정한 선형회귀식)

# In[22]:


## 만약 X(F.Coli)가 0또는 2일때 y는 몇일까 우리가 물을 떴는데 ln(F.Coli)의 값이 2였다. 그럼 몇이 나올까? 
X_new = np.array([[0], [2], [4]])
X_new_b = np.c_[np.ones((3, 1)), X_new]


# In[23]:


Y_predict = X_new_b.dot(theta_best)
Y_predict


# #### 학습과 테스트를 포함하고 있는 회귀 분석

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[27]:


len(X_train), len(X_test)


# In[30]:


X_b = np.c_[np.ones((3921, 1)), X_train]
X_b[0:5]


# In[32]:


theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y_train)
theta_best


# ### 4.0727203*1 + 0.76579042*X = Y (수학적으로 산정한 선형회귀식)

# In[35]:


X_new_b = np.c_[np.ones((1681, 1)), X_test]
Y_predict = X_new_b.dot(theta_best)
len(Y_test), len(Y_predict)


# In[40]:


MSE = (1/1681)*np.dot((Y_test - Y_predict).T , (Y_test - Y_predict))


# In[41]:


MSE # Mean Squared Error


# In[44]:


RMSE = np.sqrt(MSE)
RMSE # Root Mean Squared Error


# In[46]:


MAE = (1/1681) * sum(np.abs(Y_test - Y_predict))
MAE # Mean Absolute Error


# #### 1.2 싸이킷 럿으로 정규방정식을 사용한 선형회귀

# In[48]:


from sklearn.linear_model import LinearRegression


# In[49]:


lin_reg = LinearRegression()
lin_reg.fit(X, Y)


# In[50]:


W1 = lin_reg.intercept_
W2 = lin_reg.coef_
R_squared = lin_reg.score(X, Y)


# In[51]:


W1, W2


# ### 4.13563846*1 + 0.75613179*X = Y (수학적으로 산정한 선형회귀식)

# In[179]:


lin_reg.predict(X_new)


# In[180]:


X_lin = np.linspace(min(X), max(Y), 10000)


# In[196]:


fig, ax = plt.subplots()
ax.plot(X,Y, c = 'black',marker = '.', linestyle="") ## real data
plt.plot(X_lin, X_lin, linestyle = ':', color = 'black') ## one to one line
ax.plot(X_lin, W2*X_lin + W1, c='blue', linestyle = '--' ) ## regression line
ax.set_xlabel("ln(F.Coli)")
ax.set_ylabel("ln(T.Coli)")
ax.set_title('ln(T.coli)~ln(F.Coli)' , size = '15')
plt.text(max(X)-6, min(Y), r'$\mathrm{R}^2 = %s$' %str(round(R_squared, 2)), color = 'red')
plt.text(max(X)-6,min(Y)+2, 'Y = %s*X + %s'%(str(round(W2[0][0], 2)), str(round(W1[0], 2))), color = 'black')


# ## 2.  머신러닝을 이용한 선형 회귀 분석 

# #### 2.1 경사 하강법 (Gradient Descent)

# In[52]:


Y = np.log(new_data['t.coli'])
X = np.log(new_data['f.coli'])
X = np.array(X).reshape(len(X), 1)
Y = np.array(Y).reshape(len(Y), 1)


# In[57]:


def get_weight_updates(W2, W1, X, Y, learning_rate = 0.01):
    N = len(Y)
    ## 먼저 업데이트의 대상인 W1, W2를 각각 W1, W0의 shape와 동일한 크기의 0값 초기화 
    W1_update = np.zeros_like(W1)
    W2_update = np.zeros_like(W2)
    ## 예측배열 계산하고 예측과 실제 값의 차이 계산
    Y_pred = np.dot(X, W2.T) + W1
    diff = Y - Y_pred
    
    ## W1_update를 dot행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 형성
    W1_factors = np.ones((N, 1))
    
    #w1과 w0을 업데이트할 w1_update와 w0_update계산
    W2_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
    W1_update = -(2/N) * learning_rate *(np.dot(W1_factors.T, diff))
    
    return(W2_update, W1_update)


# In[60]:


def gradient_descent_steps(X, Y, iters = 1000):
    ## initialized number
    W1 = np.zeros((1, 1))
    W2 = np.zeros((1, 1))
    
    W1_past = []
    W2_past = []
    
    ## 반복하여 get_weight_updates()호출해 W2. W1 업데이트 
    for ind in range(iters):
        W2_update, W1_update = get_weight_updates(W2, W1, X, Y, learning_rate = 0.01)
        W2 = W2 - W2_update
        W1 = W1 - W1_update
        W1_past.append(W1)
        W2_past.append(W2)
    
    return W2, W1, W1_past, W2_past


# In[55]:


## cost function은 MSE로함
def get_cost(Y, Y_pred):
    N = len(Y)
    cost = np.sum(np.square(Y-Y_pred))/N
    return cost


# In[76]:


W2, W1, W1_past, W2_past = gradient_descent_steps(X, Y, iters = 10000)


# In[77]:


np.reshape(W1_past, (10000, 1))


# In[78]:


len(W1_past)
plt.plot(range(10000), np.reshape(W1_past, (10000, 1)))
plt.plot(range(10000), np.reshape(W2_past, (10000, 1)))


# In[295]:


print("W2 : {0:.3f}, W1 : {1:.3f}".format(W2[0, 0], W1[0, 0]))


# ### 3.813*1 + 0.799*X = Y (Gradient Descent 으로 산정한 선형회귀식)

# In[296]:


##최종적으로 예측하는 식이 이렇게 나오게 되었음. 
Y_pred = W2[0, 0]*X + W1


# In[297]:


##성능은 어떻게 됨?
print('Gradient Descent Total Cost : {0:.4f}'.format(get_cost(Y, Y_pred)))


# In[298]:


plt.scatter(X, Y)
plt.plot(X, Y_pred, 'ro')


# #### 2.2 확률적 경사하강법(Stochastic Gradient Descent)

# In[262]:


def stochastic_gradient_descent_steps(X, Y, batch_size = 10, iters = 1000):
    W2 = np.zeros((1, 1))
    W1 = np.zeros((1, 1))
    prev_cost = 100000
    iter_index = 0
    
    for ind in range(iters):
        np.random.seed(ind)
        ## 전체 X, Y데이터에서 랜덤하게 batch_size 만큼 데이터를 추출해 Sample_X, Sample_Y로 저장
        ## 조합을 랜덤하게 함.
        stochastic_random_index = np.random.permutation(X.shape[0])
        sample_X = X[stochastic_random_index[0:batch_size]]
        sample_Y = Y[stochastic_random_index[0:batch_size]]
        
        ## 랜덤하게 batch_size만큼 추출된 데이터 기반으로 w2_update, w1_update계산 후 업데이트
        W2_update, W1_update = get_weight_updates(W2, W1, sample_X, sample_Y, learning_rate = 0.0015)
        W2 = W2 - W2_update
        W1 = W1 - W1_update
        
    return W2, W1


# In[263]:


W2, W1 = stochastic_gradient_descent_steps(X, Y, iters = 10000)


# In[264]:


##### 10000번의 반복이 후 수렴한 값이 이거임
W2, W1


# In[265]:


print("W2 : {0:.3f}, W1 : {1:.3f}".format(W2[0, 0], W1[0, 0]))


# ### 4.029*1 + 0.775*X = Y (Stochastic Gradient Descent 으로 산정한 선형회귀식)

# In[266]:


Y_pred = W2[0, 0]*X + W1


# In[267]:


print('Stochastic Gradient Descent Total Cost : {0:.4f}'.format(get_cost(Y, Y_pred)))


# In[268]:


plt.scatter(X, Y)
plt.plot(X, Y_pred, 'ro')


# ### 3. 로지스틱 회귀

# In[80]:


import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[81]:


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


# In[82]:


cancer = load_breast_cancer()


# In[83]:


## 독립변수 
pd.DataFrame(cancer.data).head()


# In[84]:


## 종속변수
pd.DataFrame(cancer.target).head()


# In[85]:


## StandardScaler()로 평균이 0, 분산이 1로 데이터 분포도 변환
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)


# In[86]:


np.shape(data_scaled)


# In[87]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_scaled, cancer.target, test_size = 0.3, random_state = 0)


# In[88]:


from sklearn.metrics import accuracy_score, roc_auc_score


# In[89]:


lr_clf = LogisticRegression()
lr_clf.fit(X_train, Y_train)
lr_preds = lr_clf.predict(X_test)


# In[90]:


lr_preds[0:5]


# In[91]:


print('accuracy: {:0.3f}'.format(accuracy_score(Y_test, lr_preds)))


# In[93]:


print('roc_auc: {:0.3f}'.format(roc_auc_score(Y_test, lr_preds)))


# In[94]:


from sklearn.model_selection import GridSearchCV


# In[95]:


params = {'penalty' : ['l2', 'l1'], 'C' : [0.01, 0.1, 1, 1, 5, 10]}


# In[96]:


grid_clf = GridSearchCV(lr_clf, param_grid = params, scoring = 'accuracy', cv = 3)


# In[97]:


grid_clf.fit(data_scaled, cancer.target)


# In[98]:


print('최적 하이퍼 파라미터 : {0}, 최적 평균 정확도 :{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))

