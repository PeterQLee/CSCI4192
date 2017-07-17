import numpy as np
from scipy.integrate import odeint
from collections import deque
from models import LSTM, GRU
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_model
import sys

class MG_Series:
    tvals=np.array([])
    x_T=np.array([])


    def x_t(self,t):
        #return 1
        if t <= 0:
            return self.x0
        ind=np.searchsorted(self.tvals,t)
        if len(self.x_T)==0:return self.x0
        return self.x_T[ind-1]

    def dx_dt(self,x,t,beta,rho,tau,n):
        return beta*self.x_t(t-tau)/(1+self.x_t(t-tau)**n)-rho*x

    def generate(self,n_sets=1000):
        yl=1000
        spaces=100000
        Y=np.zeros((n_sets,spaces))
        n_params=5
        params=np.zeros((n_sets,n_params))
        dx_dt=self.dx_dt
        for i in range(n_sets):
            self.tvals=np.array([])
            self.x_T=np.array([])

            stepsize=yl/spaces
            T=np.linspace(0,yl,spaces)
            y=np.zeros(spaces)
            self.x0=np.random.uniform(low=0.1,high=10)
            x00=self.x0
            x0=self.x0
            rho=np.random.uniform(low=1,high=10)
            beta=np.random.uniform(low=1,high=10)
            tau=np.random.uniform(low=1,high=10)
            n=9.65
            params[i]=[x00,rho,beta,tau,n]
            c=0
            for t in T:
                k1=dx_dt(x0,t,beta,rho,tau,n)
                k2=dx_dt(x0+stepsize/2*k1,t+stepsize/2,beta,rho,tau,n)
                k3=dx_dt(x0+stepsize/2*k2,t+stepsize/2,beta,rho,tau,n)
                k4=dx_dt(x0+stepsize*k3,t+stepsize,beta,rho,tau,n)

                x=x0+stepsize/6*(k1+2*k2+2*k3+k4)
                self.tvals=np.concatenate((self.tvals,[t]))
                self.x_T=np.concatenate((self.x_T,[x]))
                self.x0=x
                x0=self.x0
                y[c]=x
                c+=1
            Y[i]=y

        np.save('/mnt/D2/Chaos/mg/mgset_10.npy',Y)
        np.save('/mnt/D2/Chaos/mg/params_10.npy',params)

if __name__=='__main__':
    INCREMENT=5
    if sys.argv[1]=='generate':
        MG_Series().generate(10)
    elif sys.argv[1]=='LSTM':
        #LSTM 
        data=np.load('/mnt/D2/Chaos/mg/mgset.npy')

        train=data[:300,3000:]
        test=data[970:,3000:]
        train=train[np.mean(train,axis=1)>0.1]
        test=test[np.mean(test,axis=1)>0.1]
        model=LSTM(1,1)
        trainX,trainY=train[:,:-100*INCREMENT],train[:,100*INCREMENT:]
        testX,testY=test[:,:-100*INCREMENT],test[:,100*INCREMENT:]

        err=model.train(trainX,trainY)
        
        predY=model.predict(testX)
        np.save('/mnt/D2/Chaos/mg/LSTM_eval_{}.npy'.format(INCREMENT),predY)
        np.save('/mnt/D2/Chaos/mg/LSTM_err_{}.npy'.format(INCREMENT),err)
    elif sys.argv[1]=='GRU':
        #LSTM 
        data=np.load('/mnt/D2/Chaos/mg/mgset.npy')

        train=data[:300,3000:]
        test=data[970:,3000:]
        train=train[np.mean(train,axis=1)>0.1]
        test=test[np.mean(test,axis=1)>0.1]
        model=GRU(1,1)
        trainX,trainY=train[:,:-500],train[:,500:]
        testX,testY=test[:,:-500],test[:,500:]

        err=model.train(trainX,trainY)
        

        predY=model.predict(testX)
        np.save('/mnt/D2/Chaos/mg/GRU_eval.npy',predY)
        np.save('/mnt/D2/Chaos/mg/GRU_err.npy',err)
    elif sys.argv[1]=='LSTM_long':
        data=np.load('/mnt/D2/Chaos/mg/mgset_10.npy')
        data=data[np.mean(data,axis=1)>0.1]
        ind=0
        train=data[ind,2000:5000].reshape(1,-1),data[ind,2100:5100].reshape(-1,1) #Use up to 20 to train
        test=data[ind,5500:7500].reshape(1,-1),data[ind,5600:7600].reshape(1,-1)
        #print(np.mean(train,axis=1)>0.1)
        train=train#[np.mean(train,axis=1)>0.1]
        test=test#[(np.mean(test,axis=1)>0.1)]
        model=LSTM(1,1)
        #trainX,trainY=train[:,:-500],train[:,500:]
        #ntestX,testY=test[:,:-500],test[:,500:]
        model.reset()
        for i in range(150):
            err=model.train(test[0],test[1])
            print(err)
        predY=model.predict(test[0]).flatten()

        x_t,=plt.plot(np.linspace(0,20,2000),test[0].flatten(),'--',color='green',label='x(t)')
        p_t_5,=plt.plot(np.linspace(0,20,2000),predY.flatten(),'-',label='p(t+5)',color='red')
        x_t_5,=plt.plot(np.linspace(0,20,2000),test[1].flatten(),'--',color='blue',label='x(t+5)')
        plt.legend(handles=[x_t_5,p_t_5,x_t])
        plt.show()

        #np.save('/mnt/D2/Chaos/mg/LSTM_eval.npy',predY)
        #np.save('/mnt/D2/Chaos/mg/LSTM_err.npy',err)        
    elif sys.argv[1]=='plot':
        import matplotlib.pyplot as plt
        e=np.load('/mnt/D2/Chaos/mg/LSTM_eval_{}.npy'.format(INCREMENT))
        data=np.load('/mnt/D2/Chaos/mg/mgset.npy')
        params=np.load('/mnt/D2/Chaos/mg/params.npy')
        
        test=data[970:,3000:]
        m=np.mean(test,axis=1)>0.1
        test=test[np.mean(test,axis=1)>0.1]
        import os
        os.makedirs('/mnt/D2/Chaos/mg/{}/'.format(INCREMENT),exist_ok=True)
        params=params[970:]
        params=params[m]
        for i in range(len(test)):
            length=100-INCREMENT
            plt.subplot(211)
            plt.title('x0={:.4f}, rho={:.4f}, beta={:.4f}, tau={:.4f}, n={:.4f}'.format(*params[i]))
            x_t_5,=plt.plot(np.linspace(30,length,100*(length-30)),test[i][INCREMENT*100:],'--',color='blue',label='x(t+{})'.format(INCREMENT))
            p_t_5,=plt.plot(np.linspace(30,length,100*(length-30)),e[i],'-',label='p(t+{})'.format(INCREMENT),color='red')
            plt.legend(handles=[x_t_5,p_t_5])
            plt.subplot(212)
            p_t_5,=plt.plot(np.linspace(30,length,100*(length-30)),e[i],'-',label='p(t+{})'.format(INCREMENT),color='red')
            x_5,=plt.plot(np.linspace(30,length,100*(length-30)),test[i][:-INCREMENT*100],'--',label='x(t)',color='green')
            plt.legend(handles=[p_t_5,x_5])

            #plt.show()
            plt.savefig('/mnt/D2/Chaos/mg/{}/{}.png'.format(INCREMENT,i))
            plt.clf()
    elif sys.argv[1]=='lng_LSTM':
        outdir='/mnt/D2/Chaos/mg/'
        trainx=np.load('{}/lng/trainx.npy'.format(outdir))
        trainy=np.load('{}/lng/trainy.npy'.format(outdir))
        testx=np.load('{}/lng/testx.npy'.format(outdir))
        testy=np.load('{}/lng/testy.npy'.format(outdir))

        model=LSTM(1,1)
        model.reset()
        
        
        err=model.train(trainx,trainy,ep=3)
        
        predy=model.predict(testx)

        INCREMENT=1

        np.save('{}/lng/LSTM_3.npy'.format(outdir),predy)
        np.save('{}/lng/LSTM_3err.npy'.format(outdir),err)
    elif sys.argv[1]=='lng_plot':
        outdir='/mnt/D2/Chaos/mg/'
        testx=np.load('{}/lng/testx.npy'.format(outdir))
        testy=np.load('{}/lng/testy.npy'.format(outdir))
        predy=np.load('{}/lng/LSTM_3.npy'.format(outdir))
        for i in range(len(testx)):
        
            sb=plt.subplot(211)
            #plt.title('x0={:.4f}, rho={:.4f}, beta={:.4f}, tau={:.4f}, n={:.4f}'.format(*params[i]))
            x_t_5,=plt.plot(np.linspace(0,testx[i].shape[0]//100,testx[i].shape[0]),testy[i],'--',color='blue',label='x(t+{})'.format(INCREMENT))
            p_t_5,=plt.plot(np.linspace(0,testx[i].shape[0]//100,testx[i].shape[0]),predy[i],'-',label='p(t+{})'.format(INCREMENT),color='red')
            plt.legend(handles=[x_t_5,p_t_5])
            
            sb=plt.subplot(212)
            p_t_5,=plt.plot(np.linspace(0,testx[i].shape[0]//100,testx[i].shape[0]),predy[i],'-',label='p(t+{})'.format(INCREMENT),color='red')
            x_5,=plt.plot(np.linspace(0,testx[i].shape[0]//100,testx[i].shape[0]),testx[i],'--',label='x(t)',color='green')
            plt.legend(handles=[p_t_5,x_5])
            
            plt.show()
