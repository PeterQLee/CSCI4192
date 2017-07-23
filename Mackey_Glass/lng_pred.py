from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=deque()
def x_t(tt):
    return E.pop()

def dx_dt(x,t,xt,beta=2,rho=1,tau=2,n=9.65):
    
    return beta*xt/(1+xt**n)-rho*x

def runge_kutta(x0,t_lim,n_iters,params=None):
    if params is None:
        beta=2
        rho=1
        tau=2
        n=9.65
    else:
        beta,rho,tau,n=params
    v=int((np.ceil(tau*n_iters)/t_lim).astype(np.int64))
    print(tau,v)
    E.extendleft([x0 for i in range(v)])
    T=np.linspace(0,t_lim,n_iters)
    X=np.zeros(n_iters)
    h=t_lim/n_iters
    cx=x0
    c=0
    if params is None:
        params=[beta,rho,tau,n]
    for t in T:
        xt=E.pop()
        k1=dx_dt(cx,t,xt,*params)
        k2=dx_dt(cx+k1*h/2,t+h/2,xt,*params)
        k3=dx_dt(cx+k2*h/2,t+h/2,xt,*params)
        k4=dx_dt(cx+h*k3,t+h,xt,*params)
        cx=cx+h/6*(k1+2*k2+2*k3+k4)
        E.appendleft(cx)
        X[c]=cx
        c+=1
    return T,X

def bisect_training_test(tl=2000,incrange=20,ahead=1,params=None):
    '''
    Using runge kutta method, we want to different segments of 
    training and test sets of the rk theorem.

    Will decrease with more ahead, keep this in mind.
    '''
    #x0=0.1
    x0=0.8
    ns=tl*100
    T,X=runge_kutta(x0,tl,ns,params)

    #sets will be ranges of t1-t0=20. test sets and train sets must not overlap.
    #For not no ranges will overlap
    inds=np.arange(incrange,tl,incrange)
    np.random.shuffle(inds)
    #90% train, 10% test
    
    # testX=np.zeros((len(inds)//10,(incrange-ahead)*100))
    # testY=np.zeros(testX.shape)
    # trainX=np.zeros((len(inds)-testX.shape[0],(incrange-ahead)*100))
    # trainY=np.zeros(trainX.shape)

    testX=np.zeros((len(inds)//10,(incrange)*100))
    trainX=np.zeros((len(inds)-testX.shape[0],(incrange)*100))
    
    # for i in range(len(inds)//10):
    #     testX[i]=X[inds[i]:inds[i]+(incrange-ahead)*100]
    #     testY[i]=X[inds[i]+ahead*100:inds[i]+(incrange)*100]
    # i0=i
    # for i in range(0,len(trainX),1):
    #     trainX[i]=X[inds[i+i0]:inds[i+i0]+(incrange-ahead)*100]
    #     trainY[i]=X[inds[i+i0]+ahead*100:inds[i+i0]+(incrange)*100]
    #return (trainX,trainY,testX,testY,inds)

    for i in range(len(inds)//10):
        testX[i]=X[inds[i]:inds[i]+(incrange)*100]
    i0=i
    for i in range(0,len(trainX),1):
        trainX[i]=X[inds[i+i0]:inds[i+i0]+(incrange)*100]

    return (trainX,testX,inds)

def full_set(tl=2000,incrange=20,params=None):
    x0=0.1
    ns=tl*100
    T,X=runge_kutta(x0,tl,ns,params)
    return X

if __name__=='__main__':
    import os
    #Variable beta
    outdir='/mnt/D2/Chaos/mg/lng/stable/L50/'
    os.makedirs(outdir,exist_ok=True)
    i='A'
    #for i in range(10):
    beta=np.random.uniform(low=1,high=10,size=1)
    rho=np.random.uniform(low=1,high=10,size=1)
    tau=np.random.uniform(low=1,high=10,size=1)
    n=9.65
        #params=[beta,rho,tau,n]
    params=None
    X=full_set(tl=30000,incrange=50,params=params)
    plt.plot(np.arange(3000),X[3000:6000])
    plt.savefig('{}/{}.png'.format(outdir,i))
    np.save('{}/data{}.npy'.format(outdir,i),X)
    np.save('{}/bval{}.npy'.format(outdir,i),params)
    plt.clf()
    #runge_kutta(0.1,600,60000)
    #Weird dumb paper
    # beta=0.2
    # rho=0.1
    # tau=17
    # n=10
    # train,test,inds=bisect_training_test(5500,30,0,[beta,rho,tau,n])
    
    ##Randomly generated samples
    
    # for i in range(26):
    #     ahead=4
    #     params=np.random.uniform(low=1,high=10,size=4)
    #     train,test,inds=bisect_training_test(20000,50,ahead,params)

    #     outdir='/mnt/D2/Chaos/mg/lng/rand/'

    #     CODE=chr(ord('A')+i)
    #     print(train.shape)
    #     #plt.plot(np.arange(len(train[0])),train[0])
    #     #plt.show()
    #     #ans=input('keep: ')
    #     #if ans=='y':
    #     plt.plot(np.arange(len(train[0])),train[0])
    #     plt.savefig('{}/code{}'.format(outdir,CODE))
    #     np.save('{}/train_code{}'.format(outdir,CODE),train)
    #     np.save('{}/test_code{}'.format(outdir,CODE),test)
    #     np.save('{}/inds_code{}'.format(outdir,CODE),inds)
    #     plt.clf()

    
    #np.save('{}/trainx_a{}'.format(outdir,ahead),trainx)
    #np.save('{}/trainy_a{}'.format(outdir,ahead),trainy)
    #np.save('{}/testx_a{}'.format(outdir,ahead),testx)
    #np.save('{}/testy_a{}'.format(outdir,ahead),testy)
    #np.save('{}/inds_a{}'.format(outdir,ahead),inds)

#error with exponential curve wrt time

