import numpy as np
from scipy.integrate import odeint
from collections import deque
from models import LSTM, GRU, reset_graph
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

def bisect_set(train,test,ahead,spacing=100):
    v=train.shape[1]
    trainx,trainy=train[:,:v-ahead*spacing],train[:,ahead*spacing:]
    testx,testy=test[:,:v-ahead*spacing],test[:,ahead*spacing:]
    return trainx,trainy,testx,testy

def run_models():
    base='/mnt/D2/Chaos/mg/lng/rand'
    
    for i in [1,3,5]:

        trainx=np.zeros((660,5000-i*100))
        trainy=np.zeros((660,5000-i*100))
        testx=np.zeros((660//30,5000-i*100))
        testy=np.zeros((660//30,5000-i*100))
        for j in range(0,660,30):
            #let=np.random.choice(np.arange(26))
            letter=chr(ord('A')+j//30)
            train=np.load('{}/train_code{}.npy'.format(base,letter))
            test=np.load('{}/test_code{}.npy'.format(base,letter))
            tRx,tRy,tex,tey=bisect_set(train[:30],test[:1],i)
            trainx[j:j+30]=tRx
            trainy[j:j+30]=tRy
            testx[j//30]=tex[:1]
            testy[j//30]=tey[:1]
        reset_graph()
        model=LSTM(1,1)
        err=model.train(trainx,trainy)

        predy=model.predict(testx)
        np.save('{}/testx_{}'.format(base,i),testx)
        np.save('{}/testy_{}'.format(base,i),testy)
        np.save('{}/LSTM_{}_H64.npy'.format(base,i),predy)
        np.save('{}/LSTM_{}err_H64.npy'.format(base,i),err)

        reset_graph()
        model=GRU(1,1)
        err=model.train(trainx,trainy)

        predy=model.predict(testx)
        np.save('{}/GRU_{}_H64.npy'.format(base,i),predy)
        np.save('{}/GRU_{}err_H64.npy'.format(base,i),err)


if __name__=='__main__':
    INCREMENT=5


    if sys.argv[1]=='generate':
        MG_Series().generate(10)

    if sys.argv[1]=='multi':
        run_models()
        
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
        INCREMENT=5
        outdir='/mnt/D2/Chaos/mg/'
        trainx=np.load('{}/lng/trainx_a{}.npy'.format(outdir,INCREMENT))
        trainy=np.load('{}/lng/trainy_a{}.npy'.format(outdir,INCREMENT))
        testx=np.load('{}/lng/testx_a{}.npy'.format(outdir,INCREMENT))
        testy=np.load('{}/lng/testy_a{}.npy'.format(outdir,INCREMENT))

        model=LSTM(1,1)
        model.reset()
        
        
        err=model.train(trainx,trainy,ep=1)
        
        predy=model.predict(testx)


        #print(predy.shape)
        np.save('{}/lng/LSTM_{}_H64.npy'.format(outdir,INCREMENT),predy)
        np.save('{}/lng/LSTM_{}err_H64.npy'.format(outdir,INCREMENT),err)

    elif sys.argv[1]=='lng_GRU':
        INCREMENT=3
        outdir='/mnt/D2/Chaos/mg/'
        trainx=np.load('{}/lng/trainx_a{}.npy'.format(outdir,INCREMENT))
        trainy=np.load('{}/lng/trainy_a{}.npy'.format(outdir,INCREMENT))
        testx=np.load('{}/lng/testx_a{}.npy'.format(outdir,INCREMENT))
        testy=np.load('{}/lng/testy_a{}.npy'.format(outdir,INCREMENT))

        model=GRU(1,1)
        model.reset()        
        
        err=model.train(trainx,trainy,ep=1)
        
        predy=model.predict(testx)

        np.save('{}/lng/GRU_{}_H64.npy'.format(outdir,INCREMENT),predy)
        np.save('{}/lng/GRU_{}err_H64.npy'.format(outdir,INCREMENT),err)
        
    elif sys.argv[1]=='lng_plot':
        import scipy
        INCREMENT=3
        mname='GRU'
        outdir='/mnt/D2/Chaos/mg/'
        testx=np.load('{}/lng/testx_a{}.npy'.format(outdir,INCREMENT))
        testy=np.load('{}/lng/testy_a{}.npy'.format(outdir,INCREMENT))
        #testx=np.load('{}/lng/testx_a{}.npy'.format(outdir,INCREMENT))
        #testy=np.load('{}/lng/testy_a{}.npy'.format(outdir,INCREMENT))


        predy=np.load('{}/lng/rand/{}_{}_H64.npy'.format(outdir,mname,INCREMENT))
        #predy=np.load('{}/lng/{}_{}_H64.npy'.format(outdir,mname,INCREMENT))
        #predy=np.load('{}/lng/GRU_{}_H64.npy'.format(outdir,INCREMENT))
        params=[0.1,1.0,2.0,2.0,9.65]
        style='grad'
        for i in range(len(testx)):
            
            if style=='old':
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
            elif style=='grad':
               
                dp_dt=np.gradient(predy[i].flatten())
                dy_dt=np.gradient(testy[i].flatten())
                plt.suptitle('{} delayed by {}'.format(mname,INCREMENT))
                sb=plt.subplot(221)
                plt.title('$x_0$={:.2f}, $\\rho$={:.2f}, $\\beta$={:.2f}, $\\tau$={:.2f}, n={:.2f}'.format(*params))
                
                plt.ylabel('f')
                plt.xlabel('t')
                c=20
                x_t_5,=plt.plot(np.linspace(0.,testx[i].shape[0]//100,testx[i].shape[0])[c:],testy[i][c:],'--',color='blue',label='x(t+{})'.format(INCREMENT))
                p_t_5,=plt.plot(np.linspace(0.,testx[i].shape[0]//100,testx[i].shape[0])[c:],predy[i][c:],'-',label='p(t+{})'.format(INCREMENT),color='red')
                plt.legend(handles=[x_t_5,p_t_5])
                plt.subplot(222)
                plt.ylabel('$|x-p|$'.format(INCREMENT,INCREMENT))
                plt.xlabel('t')

                plt.plot(np.linspace(0.,testx[i].shape[0]//100,testx[i].shape[0])[c:],np.abs(predy[i].flatten()[c:]-testy[i].flatten()[c:]),'--',color='blue',label='x(t+{})'.format(INCREMENT))
                plt.title('$\int |x-p| = {:.4f}$'.format(
                    np.sum(np.abs(predy[i].flatten()[c:]-testy[i].flatten()[c:]))/testx[i][c:].shape[0]))
                #plt.legend(handles=[x_t_5])
                
                sb=plt.subplot(223)
                plt.ylabel('$\\frac{df}{dt}$')
                plt.xlabel('t')
                x_t_5,=plt.plot(np.linspace(0.,testx[i].shape[0]//100,testx[i].shape[0])[c:],dy_dt[c:],'--',color='blue',label='x(t+{})'.format(INCREMENT))
                p_t_5,=plt.plot(np.linspace(0.,testx[i].shape[0]//100,testx[i].shape[0])[c:],dp_dt[c:],'-',label='p(t+{})'.format(INCREMENT),color='red')
                plt.legend(handles=[x_t_5,p_t_5])

                sb=plt.subplot(224)
                plt.ylabel('$| \\frac{dx}{dt}-\\frac{dp}{dt}|$')
                # x_t_5,=plt.plot(np.linspace(0.,testx[i].shape[0]//100,testx[i].shape[0])[c:],np.abs(dy_dt[c:]-dp_dt[c:]),'--',color='blue',label='x(t+{})'.format(INCREMENT))
                t=np.linspace(0.,testx[i].shape[0]//100,testx[i].shape[0])[c:]
                #v=np.array([0]+list(scipy.integrate.cumtrapz(np.abs(dy_dt[c:]-dp_dt[c:]),t)))
                v=np.cumsum(np.abs(dy_dt[c:]-dp_dt[c:]))/30
                print(t.shape,v.shape)
                x_t_5,=plt.plot(t,v,'--',color='blue',label='x(t+{})'.format(INCREMENT))
                
                plt.title('$\\int |\\frac{{dx}}{{dt}}-\\frac{{dp}}{{dt}}| = {:.4f}$'.format(
                    np.sum(np.abs(dy_dt[c:]-dp_dt[c:]))/30.))

                #plt.tight_layout()
                #plt.savefig('{}/lng/figures/{}_{}.png'.format(outdir,mname,INCREMENT),bbox_inches='tight')
                #break
                plt.show()
