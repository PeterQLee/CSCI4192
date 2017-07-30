import numpy as np
import matplotlib.pyplot as plt
from mg1 import bisect_set
#import seaborn as sb

def mlp_parameter(x):
    return x*(4+1+x+1+1)+1

def gru_parameter(x):
    return x*(3+3*x+3)+x*(3*x+3*x+3)+x+1

def lstm_parameter(x):
    #return 2*x*(4+4*(x)+4)+x+1
    return x*(4+4*x+4)+x*(4*x+4*x+4)+x+1

def elm_parameter(x):
    return x*(1+x+1+x+x+1+1)+1

def gradplot(predy,testy,mname,INCREMENT=1):

    for i in range(len(predy)):
        dp_dt=np.gradient(predy[i].flatten())
        dy_dt=np.gradient(testy[i].flatten())
        P=predy[i].flatten()
        T=testy[i].flatten()
        plt.suptitle('{} delayed by {}'.format(mname,INCREMENT))
        sb=plt.subplot(221)
        #plt.title('$x_0$={:.2f}, $\\rho$={:.2f}, $\\beta$={:.2f}, $\\tau$={:.2f}, n={:.2f}'.format(*params))

        plt.ylabel('f')
        plt.xlabel('t')
        c=0
        div=P[c:].shape[0]
        steps=np.linspace(500+INCREMENT,1000,T.shape[0])
        x_t_5,=plt.plot(steps[c:],T[c:],'--',color='blue',label='x(t+{})'.format(INCREMENT))
        p_t_5,=plt.plot(steps[c:],P[c:],'-',label='p(t+{})'.format(INCREMENT),color='red')
        plt.legend(handles=[x_t_5,p_t_5])
        plt.subplot(222)
        plt.ylabel('$\\int |x-p|$')
        plt.xlabel('t')

        u=np.cumsum(np.abs(T[c:]-P[c:]))/div
        plt.plot(steps[c:],u,'--',color='blue',label='x(t+{})'.format(INCREMENT))
        plt.title('$\\int |x-p| = {:.4f}$'.format(
            np.sum(np.abs(P[c:]-T[c:]))/div))
        #plt.legend(handles=[x_t_5])

        sb=plt.subplot(223)
        plt.ylabel('$\\frac{df}{dt}$')
        plt.xlabel('t')
        x_t_5,=plt.plot(steps[c:],dy_dt[c:],'--',color='blue',label='x(t+{})'.format(INCREMENT))
        p_t_5,=plt.plot(steps[c:],dp_dt[c:],'-',label='p(t+{})'.format(INCREMENT),color='red')
        plt.legend(handles=[x_t_5,p_t_5])

        sb=plt.subplot(224)
        plt.ylabel('$\\int| \\frac{dx)}{dt}-\\frac{dp}{dt}|$')
        t=steps[c:]
        v=np.cumsum(np.abs(dy_dt[c:]-dp_dt[c:]))/div
        
        x_t_5,=plt.plot(t,v,'--',color='blue',label='x(t+{})'.format(INCREMENT))

        plt.title('$\\int |\\frac{{dx}}{{dt}}-\\frac{{dp}}{{dt}}| = {:.4f}$'.format(
            np.sum(np.abs(dy_dt[c:]-dp_dt[c:]))/div))

        plt.tight_layout()
        outdir='/home/peter/Documents/CSCI4192/Chaos/figures/'
        plt.savefig('{}/{}_{}_mg2.png'.format(outdir,mname,INCREMENT),bbox_inches='tight')
        #break
        #plt.show()

def box_plot(predys,testy,mname,INCREMENT):
    M=np.zeros((3,predys[0].shape[0]))
    Md=np.zeros((3,predys[0].shape[0]))
    for m in range(3):
        #for i in range(len(predys[0])):
        dp_dt=np.gradient(predys[m].flatten())
        dy_dt=np.gradient(testy.flatten())
        P=predys[m].flatten()
        T=testy.flatten()
        div=P.shape[0]
        u=np.sum(np.abs(P-T)/div)
        v=np.sum(np.abs(dy_dt-dp_dt)/div)
        M[m,0]=u
        Md[m,0]=v
    print(M.shape)
    sb.boxplot(data=M.T)
    plt.show()
    sb.boxplot(data=Md.T)
    plt.show()
    
def scatter_plot(base,i,n_test):
    from scipy.stats import iqr
    ident='A'
    modelnames=['ELM','LSTM','GRU','MLP']
    M=np.zeros((4,4,3))
    D=np.zeros((4,4,3))
    X=np.zeros((4,4,1)) #nmodels, n_steps, data
    mp={'ELM':0,'LSTM':1,'GRU':2,'MLP':3}
    Mrange= [0.01,0.17]
    Drange=[0.004,0.03]
    hp={32:0,64:1,96:2,128:3}
    for hidden_nodes in [32,64,96,128]:
        for mname in modelnames:
            
            data=np.load('{}/data{}.npy'.format(base,ident))
            predy=np.load('{}/{}_{}_H{}.npy'.format(base,mname,i,hidden_nodes)).reshape(1,-1,1)

            testy=data[50000+i*100::100].reshape(1,-1)
            
            for C in range(9):
                py=np.load('{}/{}_{}_H{}{}.npy'.format(base,mname,i,hidden_nodes,chr(ord('a')+C))).reshape(1,-1,1)
                predy=np.concatenate((predy,py))
            print('v',predy.shape,testy.shape,base,mname,i,hidden_nodes)
            u=np.zeros(predy.shape[0])
            v=np.zeros(predy.shape[0])
            
            for j in range(predy.shape[0]):
                dp_dt=np.gradient(predy[j].flatten())
                dy_dt=np.gradient(testy[0].flatten())
                P=predy[j].flatten()
                T=testy[0].flatten()
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
    #axes=plt.gca()
    #axes.set_ylim(Mrange)
    plt.ylabel('$\\int |x - p|$')
    plt.xlabel('Number of parameters')
    plt.legend()
    plt.subplot(212)

    for d in range(len(modelnames)):
        #plt.errorbar(X[d,:,0],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=['ELM','LSTM','GRU'][d])
        plt.errorbar([uf[d](j) for j in X[d,:,0]],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    #axes=plt.gca()
    #axes.set_ylim(Drange)

    plt.legend()
    plt.ylabel('$\\int| \\frac{dx}{dt}-\\frac{dp)}{dt}|$')
    plt.xlabel('Number of parameters')
    plt.tight_layout()
    outdir='/home/peter/Documents/CSCI4192/Chaos/figures/'
    plt.savefig('{}/mg1_scatter_{}.png'.format(outdir,i))
    #plt.show()
    
def time_plot(base):
    from scipy.stats import iqr
    import seaborn as sb
    spacing=100
    ident='A'
    modelnames=['ELM','LSTM','GRU','MLP']
    M=np.zeros((len(modelnames),4,3))
    D=np.zeros((len(modelnames),4,3))
    X=np.zeros((len(modelnames),4,1)) #nmodels, n_steps, data
    mp={'ELM':0,'LSTM':1,'GRU':2,'MLP':3}
    #Mrange= [0.07,3.2]
    #Drange=[0.03,0.425]
    hp={32:0,64:1,96:2,128:3}
    tp = {1:0,5:1,10:2,30:3}

    #base='/mnt/D2/Chaos/mg/lng/beta/'

    hidden_nodes=64
    spacing=100
    for i in [1,5,10,30]:
        
        for mname in modelnames:

            data=np.load('{}/data{}.npy'.format(base,ident))
            predy=np.load('{}/{}_{}_H{}.npy'.format(base,mname,i,hidden_nodes)).reshape(1,-1,1)

            testy=data[50000+i*100::100].reshape(1,-1)
                        
            for C in range(9):
                py=np.load('{}/{}_{}_H{}{}.npy'.format(base,mname,i,hidden_nodes,chr(ord('a')+C))).reshape(1,-1,1)
                predy=np.concatenate((predy,py))

            print('v',predy.shape,testy.shape,base,mname,i,hidden_nodes)
            u=np.zeros(predy.shape[0])
            v=np.zeros(predy.shape[0])

            for j in range(predy.shape[0]):
                dp_dt=np.gradient(predy[j].flatten())
                dy_dt=np.gradient(testy[0].flatten())
                P=predy[j].flatten()
                T=testy[0].flatten()
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
    plt.subplot(212)

    for d in range(len(modelnames)):
        #plt.errorbar(X[d,:,0],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=['ELM','LSTM','GRU'][d])
        plt.errorbar([1,5,10,30],D[d,:,0],yerr=D[d,:,1:].T,fmt='--o',color=color[d],label=modelnames[d])
    #axes=plt.gca()
    #axes.set_ylim(Drange)

    plt.legend()
    plt.ylabel('$\\int| \\frac{dx}{dt}-\\frac{dp)}{dt}|$')
    plt.xlabel('$t_0$')
    plt.tight_layout()
    plt.suptitle('Performance decay with increased $t_0$')
    outdir='/Users/Peter/Documents/CSCI4192/Chaos/figures/'
    plt.savefig('{}/mg1_save.png'.format(outdir))
    #plt.show()

    
if __name__=='__main__':
    incrange=50
    hidden_nodes=64
    #base='/mnt/D2/Chaos/mg/lng/stable2/'
    base='/Users/Peter/Documents/CSCI4192/Chaos/stable2'
    ident='A'

    spacing=100
    data=np.load('{}/data{}.npy'.format(base,ident))
    i=1
    mname='LSTM'
    predy=np.load('{}/{}_{}_H{}a.npy'.format(base,mname,i,hidden_nodes))
    
    #predy=np.transpose(predy,[1,2,0])
    #data=data[:
    
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
    time_plot(base)
