import numpy as np
from eval_mg1 import gradplot, elm_parameter, gru_parameter, lstm_parameter
from mg1 import bisect_set
import matplotlib.pyplot as plt

def scatter_plot(base,i,n_test):
    from scipy.stats import iqr
    spacing=100
    ident='A'
    modelnames=['ELM','LSTM','GRU']
    M=np.zeros((len(modelnames),4,3))
    D=np.zeros((len(modelnames),4,3))
    X=np.zeros((len(modelnames),4,1)) #nmodels, n_steps, data
    mp={'ELM':0,'LSTM':1,'GRU':2}#,'MLP':3}
    #Mrange= [0.07,3.2]
    #Drange=[0.03,0.425]
    hp={32:0,64:1,96:2,128:3}

    base='/mnt/D2/Chaos/mg/lng/beta/'
    train_data=[]
    test_data=[]

    for j in range(0,10,1):
        ident=chr(ord('K')+j)
        data=np.load('{}/data{}.npy'.format(base,ident))
        train=data[:len(data)//2]
        test=data[len(data)//2:]
        train=train[::spacing]
        test=test[::spacing]
        train=train.reshape(1,-1)
        test=test.reshape(1,-1)
        train_data.append(train)
        test_data.append(test)

    for hidden_nodes in [32,64,96,128]:
        for mname in modelnames:
            
            #data=np.load('{}/data{}.npy'.format(base,ident))
            #predy=np.load('{}/{}_{}_H{}.npy'.format(base,mname,i,hidden_nodes)).reshape(1,-1,1)

            #testy=data[50000+i*100::100].reshape(1,-1)
            predy=np.array([]).reshape(-1,train_data[0].shape[1]-i)
            testy=np.array([]).reshape(-1,train_data[0].shape[1]-i)
            for C in range(10):
                ident=chr(ord('K')+C)
                #dy=np.load('{}/data{}.npy'.format(base,ident)).reshape(1,-1,1)
                dy=test_data[C].reshape(1,-1)[:,i:]
                py=np.load('{}/{}_{}_H{}{}.npy'.format(base,mname,i,hidden_nodes,chr(ord('a')+C))).reshape(1,-1)
                predy=np.concatenate((predy,py))
                print(dy.shape)
                testy=np.concatenate((testy,dy))
                
            print('v',predy.shape,testy.shape,base,mname,i,hidden_nodes)
            u=np.zeros(predy.shape[0])
            v=np.zeros(predy.shape[0])
            
            for j in range(predy.shape[0]):
                dp_dt=np.gradient(predy[j].flatten())
                dy_dt=np.gradient(testy[j].flatten())
                P=predy[j].flatten()
                T=testy[j].flatten()
                div=P.shape[0]
                u[j]=np.sum(np.abs(P-T)/div)
                v[j]=np.sum(np.abs(dy_dt-dp_dt)/div)
            
            X[mp[mname],hp[hidden_nodes],:]=hidden_nodes
            M[mp[mname],hp[hidden_nodes],0]=np.median(u)
            M[mp[mname],hp[hidden_nodes],1]=iqr(u,rng=(25,50))
            M[mp[mname],hp[hidden_nodes],2]=iqr(u,rng=(50,75))
            D[mp[mname],hp[hidden_nodes],0]=np.median(v)
            D[mp[mname],hp[hidden_nodes],1]=iqr(v,rng=(25,50))
            D[mp[mname],hp[hidden_nodes],2]=iqr(v,rng=(50,75))
    d=0
    color=['blue','green','red','orange']
    uf=(elm_parameter,lstm_parameter,gru_parameter)
    plt.suptitle('Forward prediction by {}'.format(i))
    plt.subplot(211)
    for d in range(len(modelnames)):
        print(M[d,:,0],M[d,:,1])
        plt.errorbar([uf[d](j) for j in X[d,:,0]],M[d,:,0],yerr=M[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    #axes=plt.gca()
    #axes.set_ylim(Mrange)
    plt.ylabel('$\\int |x - p|$')
    plt.xlabel('Number of parameters')
    plt.legend()
    plt.grid(True)
    plt.subplot(212)

    for d in range(len(modelnames)):
        #plt.errorbar(X[d,:,0],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=['ELM','LSTM','GRU'][d])
        plt.errorbar([uf[d](j) for j in X[d,:,0]],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    #axes=plt.gca()
    #axes.set_ylim(Drange)

    plt.legend()
    plt.ylabel('$\\int| \\frac{dx}{dt}-\\frac{dp)}{dt}|$')
    plt.xlabel('Number of parameters')
    plt.grid(True)
    plt.tight_layout()
    outdir='/home/peter/Documents/CSCI4192/Chaos/figures/'
    plt.savefig('{}/mg2_scatter_{}.png'.format(outdir,i))
    #plt.show()

def ttest(base,i,n_test):
    from scipy.stats import iqr, ttest_1samp
    spacing=100
    ident='A'
    modelnames=['ELM','LSTM','GRU']
    M=np.zeros((len(modelnames),4,10))
    D=np.zeros((len(modelnames),4,10))
    X=np.zeros((len(modelnames),4,1)) #nmodels, n_steps, data
    mp={'ELM':0,'LSTM':1,'GRU':2}#,'MLP':3}
    #Mrange= [0.07,3.2]
    #Drange=[0.03,0.425]
    hp={32:0,64:1,96:2,128:3}
    time_map={1:0,5:1,10:2,30:3}
    z=0
    base='/mnt/D2/Chaos/mg/lng/beta/'
    train_data=[]
    test_data=[]

    resM=np.zeros((len(modelnames),4,3)) #models,timesteps,median,q-,q+
    resD=np.zeros((len(modelnames),4,3)) #models,timesteps,median,q-,q+

    
    for i in [1,5,10,30]:
        for j in range(0,10,1):
            ident=chr(ord('K')+j)
            data=np.load('{}/data{}.npy'.format(base,ident))
            train=data[:len(data)//2]
            test=data[len(data)//2:]
            train=train[::spacing]
            test=test[::spacing]
            train=train.reshape(1,-1)
            test=test.reshape(1,-1)
            train_data.append(train)
            test_data.append(test)

        for hidden_nodes in [32,64,96,128]:
            for mname in modelnames:

                #data=np.load('{}/data{}.npy'.format(base,ident))
                #predy=np.load('{}/{}_{}_H{}.npy'.format(base,mname,i,hidden_nodes)).reshape(1,-1,1)

                #testy=data[50000+i*100::100].reshape(1,-1)
                predy=np.array([]).reshape(-1,train_data[0].shape[1]-i)
                testy=np.array([]).reshape(-1,train_data[0].shape[1]-i)
                for C in range(10):
                    ident=chr(ord('K')+C)
                    #dy=np.load('{}/data{}.npy'.format(base,ident)).reshape(1,-1,1)
                    dy=test_data[C].reshape(1,-1)[:,i:]
                    py=np.load('{}/{}_{}_H{}{}.npy'.format(base,mname,i,hidden_nodes,chr(ord('a')+C))).reshape(1,-1)
                    predy=np.concatenate((predy,py))

                    testy=np.concatenate((testy,dy))

                #print('v',predy.shape,testy.shape,base,mname,i,hidden_nodes)
                u=np.zeros(predy.shape[0])
                v=np.zeros(predy.shape[0])

                for j in range(predy.shape[0]):
                    dp_dt=np.gradient(predy[j].flatten())
                    dy_dt=np.gradient(testy[j].flatten())
                    P=predy[j].flatten()
                    T=testy[j].flatten()
                    div=P.shape[0]
                    u[j]=np.sum(np.abs(P-T)/div)
                    v[j]=np.sum(np.abs(dy_dt-dp_dt)/div)

                #X[mp[mname],hp[hidden_nodes],:]=hidden_nodes
                M[mp[mname],hp[hidden_nodes],:]=u

                D[mp[mname],hp[hidden_nodes],:]=v
        h=[np.argmin(np.median(M[a],axis=1)) for a in range(3)]
        import scipy.stats as stats
        med=[np.median(M[a,h[a]]) for a in range(3)]
        print(i,'&'.join(['{:.4f} $\pm$ {:.4f} '.format(med[a],stats.iqr(M[a,h[a]])) for a in range(3)]))
        medD=[np.median(D[a,h[a]]) for a in range(3)]
        mlow=[stats.iqr(M[a,h[a]],rng=(25,50)) for a in range(3)]
        mhigh=[stats.iqr(M[a,h[a]],rng=(50,75)) for a in range(3)]
        dlow=[stats.iqr(D[a,h[a]],rng=(25,50)) for a in range(3)]
        dhigh=[stats.iqr(D[a,h[a]],rng=(50,75)) for a in range(3)]

        resM[:,time_map[i],0]=med
        resM[:,time_map[i],1]=mlow
        resM[:,time_map[i],2]=mhigh
        resD[:,time_map[i],0]=medD
        resD[:,time_map[i],1]=dlow
        resD[:,time_map[i],2]=dhigh

    plt.subplot(211)
    plt.grid(True)
    color=['blue','green','red','orange']
    for d in range(len(modelnames)):
        plt.errorbar([1,5,10,30],resM[d,:,0],yerr=resM[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    plt.ylabel('$\\int |x - p|$')
    plt.xlabel('$t_0$')
    plt.legend()
    plt.subplot(212)
    plt.grid(True)
    for d in range(len(modelnames)):
        plt.errorbar([1,5,10,30],resD[d,:,0],yerr=resD[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    #axes=plt.gca()
    #axes.set_ylim(Drange)
    plt.legend()
    plt.ylabel('$\\int| \\frac{dx}{dt}-\\frac{dp)}{dt}|$')
    plt.xlabel('$t_0$')
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.suptitle('Performance with respect to $t_0$')
    outdir='/home/peter/Documents/CSCI4192/Chaos/figures/'
    #plt.show()
    plt.savefig('{}/mg2_save.png'.format(outdir))

def time_plot(base):
    from scipy.stats import iqr

    
    spacing=100
    ident='A'
    modelnames=['ELM','LSTM','GRU']
    M=np.zeros((len(modelnames),4,3))
    D=np.zeros((len(modelnames),4,3))
    X=np.zeros((len(modelnames),4,1)) #nmodels, n_steps, data
    mp={'ELM':0,'LSTM':1,'GRU':2}#,'MLP':3}
    #Mrange= [0.07,3.2]
    #Drange=[0.03,0.425]
    hp={32:0,64:1,96:2,128:3}
    tp = {1:0,5:1,10:2,30:3}

    #base='/mnt/D2/Chaos/mg/lng/beta/'

    hidden_nodes=64
    spacing=100
    for i in [1,5,10,30]:
        train_data=[]
        test_data=[]

        for j in range(0,10,1):
            ident=chr(ord('K')+j)
            data=np.load('{}/data{}.npy'.format(base,ident))
            train=data[:len(data)//2]
            test=data[len(data)//2:]
            train=train[::spacing]
            test=test[::spacing]
            train=train.reshape(1,-1)
            test=test.reshape(1,-1)
            train_data.append(train)
            test_data.append(test)
        for mname in modelnames:

            #data=np.load('{}/data{}.npy'.format(base,ident))
            #predy=np.load('{}/{}_{}_H{}.npy'.format(base,mname,i,hidden_nodes)).reshape(1,-1,1)

            #testy=data[50000+i*100::100].reshape(1,-1)
            predy=np.array([]).reshape(-1,train_data[0].shape[1]-i)
            testy=np.array([]).reshape(-1,train_data[0].shape[1]-i)
            for C in range(10):
                ident=chr(ord('K')+C)
                #dy=np.load('{}/data{}.npy'.format(base,ident)).reshape(1,-1,1)
                dy=test_data[C].reshape(1,-1)[:,i:]
                py=np.load('{}/{}_{}_H{}{}.npy'.format(base,mname,i,hidden_nodes,chr(ord('a')+C))).reshape(1,-1)
                print(py.shape,predy.shape)
                predy=np.concatenate((predy,py))
                print(dy.shape)
                testy=np.concatenate((testy,dy))

            print('v',predy.shape,testy.shape,base,mname,i,hidden_nodes)
            u=np.zeros(predy.shape[0])
            v=np.zeros(predy.shape[0])

            for j in range(predy.shape[0]):
                dp_dt=np.gradient(predy[j].flatten())
                dy_dt=np.gradient(testy[j].flatten())
                P=predy[j].flatten()
                T=testy[j].flatten()
                div=P.shape[0]
                u[j]=np.sum(np.abs(P-T)/div)
                v[j]=np.sum(np.abs(dy_dt-dp_dt)/div)

            X[mp[mname],tp[i],:]=hidden_nodes
            M[mp[mname],tp[i],0]=np.median(u)
            M[mp[mname],tp[i],1]=iqr(u,rng=(25,50))
            M[mp[mname],tp[i],2]=iqr(u,rng=(50,75))
            D[mp[mname],tp[i],0]=np.median(v)
            D[mp[mname],tp[i],1]=iqr(v,rng=(25,50))
            D[mp[mname],tp[i],2]=iqr(v,rng=(50,75))

    plt.subplot(211)
    color=['blue','green','red','orange']
    for d in range(len(modelnames)):
        print(M[d,:,0],M[d,:,1])
        plt.errorbar([1,5,10,30],M[d,:,0],yerr=M[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    #axes=plt.gca()
    #axes.set_ylim(Mrange)
    plt.ylabel('$\\int |x - p|$')
    plt.xlabel('$t_0$')
    plt.legend()
    plt.grid(True)
    plt.subplot(212)

    for d in range(len(modelnames)):
        #plt.errorbar(X[d,:,0],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=['ELM','LSTM','GRU'][d])
        plt.errorbar([1,5,10,30],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    #axes=plt.gca()
    #axes.set_ylim(Drange)

    plt.legend()
    plt.ylabel('$\\int| \\frac{dx}{dt}-\\frac{dp)}{dt}|$')
    plt.xlabel('$t_0$')
    plt.grid(True)
    plt.tight_layout()
    plt.suptitle('Performance decay with increased $t_0$')
    outdir='/home/peter/Documents/CSCI4192/Chaos/figures/'
    plt.savefig('{}/mg2_save.png'.format(outdir))
    #plt.show()
    
if __name__=='__main__':
    incrange=50
    hidden_nodes=32
    base='/mnt/D2/Chaos/mg/lng/beta/'
    #base='/Users/Peter/Documents/CSCI4192/Chaos/beta/'
    ident='M'

    spacing=100
    data=np.load('{}/data{}.npy'.format(base,ident))
    i=30
    mname='LSTM'
    predy=np.load('{}/{}_{}_H{}c.npy'.format(base,mname,i,hidden_nodes))

    
    train,test=bisect_set(data,i,incrange,spacing)
    testy=test[:,100*i:]
    
    ############
    testy=data[50000+100*i::100].reshape(1,-1)
    
    ############
    print(testy.shape,predy.shape)
    predy=predy.reshape(1,-1,1)
    #gradplot(predy,testy,mname,i)

    #P=(np.load('{}/ELM_{}_H{}.npy'.format(base,i,hidden_nodes)),np.load('{}/LSTM_{}_H{}.npy'.format(base,i,hidden_nodes)),np.load('{}/GRU_{}_H{}.npy'.format(base,i,hidden_nodes)))

    #box_plot(P,testy,mname,i)
    print(predy.shape)
    
    #scatter_plot(base,i,predy.shape[0])
    #time_plot(base)
    ttest(base,i,predy.shape[0])
