import numpy as np
from scipy.integrate import odeint
from collections import deque
from models import LSTM, GRU
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_model

tvals=np.array([])
x_T=np.array([])
x0=0.1

def x_t(t):
    #return 1
    if t <= 0:
        return x0
    ind=np.searchsorted(tvals,t)
    return x_T[ind-1]

def dx_dt(x,t,beta,rho,tau,n):
    return beta*x_t(t-tau)/(1+x_t(t-tau)**n)-rho*x
yl=100
spaces=10000
stepsize=yl/spaces
T=np.linspace(0,yl,spaces)
y=np.zeros(spaces)
x0=np.random.uniform(low=0.1,high=5)
x00=x0
rho=np.random.uniform(low=0,high=5)
beta=np.random.uniform(low=0,high=5)
tau=np.random.uniform(low=0,high=5)
n=9.65
c=0
for t in T:
    k1=dx_dt(x0,t,beta,rho,tau,n)
    k2=dx_dt(x0+stepsize/2*k1,t+stepsize/2,beta,rho,tau,n)
    k3=dx_dt(x0+stepsize/2*k2,t+stepsize/2,beta,rho,tau,n)
    k4=dx_dt(x0+stepsize*k3,t+stepsize,beta,rho,tau,n)

    x=x0+stepsize/6*(k1+2*k2+2*k3+k4)
    tvals=np.concatenate((tvals,[t]))
    x_T=np.concatenate((x_T,[x]))
    x0=x
    y[c]=x
    c+=1

plt.title('x0={:.4f} rho={:.4f}, beta={:.4f} tau={:.4f}'.format(x00,rho,beta,tau))
plt.plot(T,y)
plt.show()


#Training
# yl=100
# spaces=10000
# T=np.linspace(0,yl,spaces)

# model=LSTM(1,1)
# traindata=[]
# for i in range(100):
#     tvals=np.array([])
#     x_T=np.array([])

#     y=np.zeros(spaces)
#     x0=0.1
#     rho=np.random.uniform(low=0,high=5)
#     beta=np.random.uniform(low=0,high=5)
#     tau=np.random.uniform(low=0,high=5)
#     n=9.65
#     c=0
#     for t in T:
#         x=x0+yl/spaces*dx_dt(x0,beta,rho,tau,n)
#         #D[t]=x
#         tvals=np.concatenate((tvals,[t]))
#         x_T=np.concatenate((x_T,[x]))
#         x0=x
#         y[c]=x
#         c+=1

#     p=y[:-100]
#     l=y[100:]

#     traindata.append((p,l))

# model.train(traindata)

# for i in range(3):
#     tvals=np.array([])
#     x_T=np.array([])

#     y=np.zeros(spaces)
#     x0=0.1
#     rho=np.random.uniform(low=0,high=5)
#     beta=np.random.uniform(low=0,high=5)
#     tau=np.random.uniform(low=0,high=5)
#     n=9.65
#     c=0
#     for t in T:
#         x=x0+yl/spaces*dx_dt(x0,beta,rho,tau,n)
#         #D[t]=x
#         tvals=np.concatenate((tvals,[t]))
#         x_T=np.concatenate((x_T,[x]))
#         x0=x
#         y[c]=x
#         c+=1
#     p=y[:-100]
#     l=y[100:]
#     pred=model.predict(p)
#     plt.plot(np.arange(spaces-100),l,'--')
#     plt.plot(np.arange(spaces-100),pred.flatten(),'-')
#     plt.show()
