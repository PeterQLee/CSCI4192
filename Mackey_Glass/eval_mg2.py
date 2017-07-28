import numpy as np
from eval_mg1 import gradplot
from mg1 import bisect_set


def scatter_plot(base,i,n_test):
    from scipy.stats import iqr
    spacing=100
    ident='A'
    modelnames=['ELM','LSTM','GRU','MLP']
    M=np.zeros((4,4,3))
    D=np.zeros((4,4,3))
    X=np.zeros((4,4,1)) #nmodels, n_steps, data
    mp={'ELM':0,'LSTM':1,'GRU':2,'MLP':3}
    Mrange= [0.01,0.17]
    Drange=[0.004,0.03]
    hp={32:0,64:1,96:2,128:3}

    base='/mnt/D2/Chaos/mg/lng/beta/'
    train_data=[]
    test_data=[]
    for i in range(0,10,1):
        ident=chr(ord('K')+i)
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
            predy=np.reshape(-1,train_data[0].shape[1])
            testy=np.reshape(-1,train_data[0].shape[1])
            for C in range(10):
                ident=chr(ord('K')+C)
                dy=np.load('{}/data{}.npy'.format(base,ident))
                py=np.load('{}/{}_{}_H{}{}.npy'.format(base,mname,i,hidden_nodes,chr(ord('a')+C))).reshape(1,-1,1)
                predy=np.concatenate((predy,py))
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
    uf=(elm_parameter,lstm_parameter,gru_parameter,mlp_parameter)
    plt.suptitle('Forward prediction by {}'.format(i))
    plt.subplot(211)
    for d in range(len(modelnames)):
        print(M[d,:,0],M[d,:,1])
        plt.errorbar([uf[d](j) for j in X[d,:,0]],M[d,:,0],yerr=M[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    axes=plt.gca()
    axes.set_ylim(Mrange)
    plt.ylabel('$\\int |x - p|$')
    plt.xlabel('Number of parameters')
    plt.legend()
    plt.subplot(212)

    for d in range(len(modelnames)):
        #plt.errorbar(X[d,:,0],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=['ELM','LSTM','GRU'][d])
        plt.errorbar([uf[d](j) for j in X[d,:,0]],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    axes=plt.gca()
    axes.set_ylim(Drange)

    plt.legend()
    plt.ylabel('$\\int| \\frac{dx}{dt}-\\frac{dp)}{dt}|$')
    plt.xlabel('Number of parameters')
    plt.tight_layout()
    outdir='/home/peter/Documents/CSCI4192/Chaos/figures/'
    plt.savefig('{}/mg1_scatter_{}.png'.format(outdir,i))

    
if __name__=='__main__':
    incrange=50
    hidden_nodes=32
    base='/mnt/D2/Chaos/mg/lng/beta/'
    ident='M'

    spacing=100
    data=np.load('{}/data{}.npy'.format(base,ident))
    i=1
    mname='MLP'
    predy=np.load('{}/{}_{}_H{}c.npy'.format(base,mname,i,hidden_nodes))

    
    train,test=bisect_set(data,i,incrange,spacing)
    testy=test[:,100*i:]
    
    ############
    testy=data[50000+100*i::100].reshape(1,-1)
    
    ############
    print(testy.shape,predy.shape)
    predy=predy.reshape(1,-1,1)
    gradplot(predy,testy,mname,i)

    #P=(np.load('{}/ELM_{}_H{}.npy'.format(base,i,hidden_nodes)),np.load('{}/LSTM_{}_H{}.npy'.format(base,i,hidden_nodes)),np.load('{}/GRU_{}_H{}.npy'.format(base,i,hidden_nodes)))

    #box_plot(P,testy,mname,i)
    print(predy.shape)
    
    scatter_plot(base,i,predy.shape[0])
