import matplotlib.pyplot as plt
import numpy as np


'''
1. 플랏 만들기
'''

# 단일 플랏 만들기

fig, ax = plt.subplots()

ax.plot([1,2,3], [1,2,3])

#plt.show()

# 복수 플랏 만들기
fig, (ax1, ax2) = plt.subplots(nrows = 1,ncols = 2)

ax1.plot( np.random.rand(2), marker = 'o' , linestyle = '' )
ax2.plot(np.random.rand(10), marker = '^', linestyle = '', color = 'red')

#plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)

ax1.plot( np.random.rand(2), marker = 'o' , linestyle = '' )
ax2.plot( np.random.rand(4), marker = 'o' , linestyle = '' )
ax3.plot( np.random.rand(5), marker = 'o' , linestyle = '' )
ax4.plot( np.random.rand(6), marker = 'o' , linestyle = '' )
#plt.show()

fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[0,0].plot(np.random.rand(4), marker = 'o' , linestyle = '' )
ax[0,1].plot(np.random.rand(5), marker = 'o' , linestyle = '' )
ax[1,0].plot(np.random.rand(6), marker = 'o' , linestyle = '' )
ax[1,1].plot(np.random.rand(7), marker = 'o' , linestyle = '' )
#plt.show()


fig, ax = plt.subplots(nrows = 10, ncols = 10)
for i in range(len(list(range(1, 11, 1)))):
    for j in range(len(list(range(1, 11, 1)))):
        ax[i, j].plot(np.random.rand(i + j),np.random.rand(i + j), marker ='o', linestyle = '')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
#plt.show()



'''
2. Line plot

'''

fig, ax = plt.subplots(nrows = 1, ncols = 1)

import random

x = [ random.randint(0, 100) for i in range(100) ]
x.sort()
y = [ random.randint(0, 100) for i in range(100) ]
y.sort()

ax.plot(x, y)

## 참고 : scale변환
### 'linear', 'log', 'symlog', 'logit'...
ax.set_xscale('log')
ax.set_yscale('log')


plt.show()


'''
3. Scatter plot

'''
fig, ax = plt.subplots(nrows = 1, ncols = 1)
import random
x = [ random.randint(0, 100) for i in range(100) ]
x.sort()
y = [ random.randint(0, 100) for i in range(100) ]
y.sort()

ax.scatter(x, y, marker = 'D',  color = '#ffcccc')
ax.scatter(x, y, marker = 'D', color = (244/256, 255/256, 123/256))

plt.show()


## 참고 : 색깔 변환
# rgb의 경우 튜플형식으로 대입 0과 1사이로 만약 244, 255, 123 이면 256으로 나눈 수가 안에 들어감 ( , , )
# html의 경우 7자리 코드로 대입


'''
Q1 x와 y의 np.random.normal(mu = 10, sigma = 1,size = 100)으로 100개씩 난수 추출
Q2 x와 y의 np.random.poisson(lam = 20, size = 100)으로 100개씩 난수 추출 

'''


'''
4. Histogram
'''
fig, ax = plt.subplots(nrows = 1, ncols = 1)
import random
x = [ random.randint(0, 100) for i in range(100) ]
x.sort()
y = [ random.randint(0, 100) for i in range(100) ]
y.sort()


#ax.hist(x, bins = 15)

ax.hist(x,
         bins = 10,
         density = False,
         cumulative = False,
         histtype = 'bar', # bar or step
         orientation = 'vertical', # or horizontal
         rwidth = 0.8, ## 1.0일 경우, 꽉 채움 작아질수록 간격이 생김
         color = 'red')


### 참고 : 텍스트 삽입
ax.text(50, 6, "text",
        fontsize = 20,
        rotation = 60)

plt.show()


'''
5. 라벨 붙이기 x축이름 y축이름 제목
'''
fig, ax = plt.subplots(nrows = 1, ncols = 1)
x = np.random.normal(10, 3,size = 100)
y = np.random.normal(4, 1, size = 100)

ax.plot(x, y, 'o')

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_title("x, y graph")

plt.show()

'''
Q3 x의 히스토그램을 만들고 x, y축이름과 제목을 붙여보시오

'''



'''
6. Ticks
Ticks의 첫번째 파라미터는 수치 리스트가 들어가고, 두번째 파라미터는 첫번째 파라미터를
각각 대체하는 역할을 한다.
'''
fig, ax = plt.subplots(nrows = 1, ncols = 1)

import random

x = [ random.randint(0, 100) for i in range(100) ]
x.sort()
y = [ random.randint(0, 100) for i in range(100) ]
y.sort()
scale = 20 * np.random.randint(1, 10, size = 100)
c = np.random.randint(1, 10, size=100)


ax.set_ylim(10, 120)
ax.set_xlim(10, 120)

ax.set_xscale('log')
ax.set_yscale('log')

ax.scatter(x, y, s = scale, marker = 'o', c = c, label = 'whatis!', alpha = 0.6)

tick_val = (1, 10, 100)
tick_lab = ('one', 'ten', 'hundred')
ax.set_xticks(tick_val)
ax.set_xticklabels(tick_lab)

ax.set_yticks(tick_val)
ax.set_yticklabels(tick_lab)

ax.set_ylabel("ylabel")
ax.set_xlabel("xlabel")
ax.set_title("x, y")

ax.legend(loc = "upper left")
ax.grid(True)

plt.show()


'''
Q4 X, Y 난수 100개를 만들고 그림을 그려보시오

제목 : X, Y graph
x 제목 : X-axis
y 제목 : Y-axis

'''






