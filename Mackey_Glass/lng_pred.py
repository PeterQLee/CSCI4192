from collections import deque
import numpy as np
import matplotlib.pyplot as plt
E=deque()
def x_t(tt):
    return E.pop()

def dx_dt(x,t,xt,beta=2,rho=1,tau=2,n=9.65):
    
    return beta*xt/(1+xt**n)-rho*x

def runge_kutta(x0,t_lim,n_iters):
    beta=2
    rho=1
    tau=2
    n=9.65
    
    E.extendleft([x0 for i in range(tau*n_iters//t_lim)])
    T=np.linspace(0,t_lim,n_iters)
    X=np.zeros(n_iters)
    h=t_lim/n_iters
    cx=x0
    c=0
    for t in T:
        xt=E.pop()
        k1=dx_dt(cx,t,xt)
        k2=dx_dt(cx+k1*h/2,t+h/2,xt)
        k3=dx_dt(cx+k2*h/2,t+h/2,xt)
        k4=dx_dt(cx+h*k3,t+h,xt)
        cx=cx+h/6*(k1+2*k2+2*k3+k4)
        E.appendleft(cx)
        X[c]=cx
        c+=1
    return T,X
def bisect_training_test(tl=2000,incrange=20,ahead=1):
    '''
    Using runge kutta method, we want to different segments of 
    training and test sets of the rk theorem.

    Will decrease with more ahead, keep this in mind.
    '''
    x0=0.1
    ns=tl*100
    T,X=runge_kutta(x0,tl,ns)

    #sets will be ranges of t1-t0=20. test sets and train sets must not overlap.
    #For not no ranges will overlap
    inds=np.arange(incrange,tl,incrange)
    np.random.shuffle(inds)
    #90% train, 10% test
    
    testX=np.zeros((len(inds)//10,(incrange-ahead)*100))
    testY=np.zeros(testX.shape)
    trainX=np.zeros((len(inds)-testX.shape[0],(incrange-ahead)*100))
    trainY=np.zeros(trainX.shape)

    for i in range(len(inds)//incrange):
        testX[i]=X[inds[i]:inds[i]+(incrange-ahead)*100]
        testY[i]=X[inds[i]+ahead*100:inds[i]+(incrange)*100]
    for i in range(0,int(tl-incrange)//incrange,len(inds)):
        trainX[i]=X[inds[i]:inds[i]+(incrange-ahead)*100]
        trainY[i]=X[inds[i]+ahead*100:inds[i]+(incrange)*100]
    return (trainX,trainY,testX,testY,inds)
if __name__=='__main__':
    
    #runge_kutta(0.1,600,60000)
    
    trainx,trainy,testx,testy,inds=bisect_training_test(20000,30)
    
    #print(a.shape,b.shape,c.shape,d.shape)
    #plt.plot(np.linspace(0,len(a[0])//100,len(a[0])),a[0])
    #plt.plot(np.linspace(0,len(a[0])//100,len(a[0])),b[0])
    #plt.show()
    outdir='/mnt/D2/Chaos/mg/lng/'
    np.save('{}/trainx'.format(outdir),trainx)
    np.save('{}/trainy'.format(outdir),trainy)
    np.save('{}/testx'.format(outdir),testx)
    np.save('{}/testy'.format(outdir),testy)
    np.save('{}/inds'.format(outdir),inds)
    
#error with exponential curve wrt time
