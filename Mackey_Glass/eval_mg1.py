import numpy as np
import matplotlib.pyplot as plt
from mg1 import bisect_set
import seaborn as sb

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
        
        x_t_5,=plt.plot(np.linspace(0.,T.shape[0]//100,T.shape[0])[c:],T[c:],'--',color='blue',label='x(t+{})'.format(INCREMENT))
        p_t_5,=plt.plot(np.linspace(0.,T.shape[0]//100,T.shape[0])[c:],P[c:],'-',label='p(t+{})'.format(INCREMENT),color='red')
        plt.legend(handles=[x_t_5,p_t_5])
        plt.subplot(222)
        plt.ylabel('$|x-p|$'.format(INCREMENT,INCREMENT))
        plt.xlabel('t')

        u=np.cumsum(np.abs(T[c:]-P[c:]))/div
        plt.plot(np.linspace(0.,T.shape[0]//100,T.shape[0])[c:],u,'--',color='blue',label='x(t+{})'.format(INCREMENT))
        plt.title('$\int |x-p| = {:.4f}$'.format(
            np.sum(np.abs(P[c:]-T[c:]))/div))
        #plt.legend(handles=[x_t_5])

        sb=plt.subplot(223)
        plt.ylabel('$\\frac{df}{dt}$')
        plt.xlabel('t')
        x_t_5,=plt.plot(np.linspace(0.,T.shape[0]//100,T.shape[0])[c:],dy_dt[c:],'--',color='blue',label='x(t+{})'.format(INCREMENT))
        p_t_5,=plt.plot(np.linspace(0.,T.shape[0]//100,T.shape[0])[c:],dp_dt[c:],'-',label='p(t+{})'.format(INCREMENT),color='red')
        plt.legend(handles=[x_t_5,p_t_5])

        sb=plt.subplot(224)
        plt.ylabel('$| \\frac{dx}{dt}-\\frac{dp}{dt}|$')
        t=np.linspace(0.,T.shape[0]//100,T.shape[0])[c:]
        v=np.cumsum(np.abs(dy_dt[c:]-dp_dt[c:]))/div
        
        x_t_5,=plt.plot(t,v,'--',color='blue',label='x(t+{})'.format(INCREMENT))

        plt.title('$\\int |\\frac{{dx}}{{dt}}-\\frac{{dp}}{{dt}}| = {:.4f}$'.format(
            np.sum(np.abs(dy_dt[c:]-dp_dt[c:]))/div))

        #plt.tight_layout()
        #plt.savefig('{}/lng/figures/{}_{}.png'.format(outdir,mname,INCREMENT),bbox_inches='tight')
        #break
        plt.show()

def box_plot(predys,testy,mname,INCREMENT):
    M=np.zeros((3,predys[0].shape[0]))
    Md=np.zeros((3,predys[0].shape[0]))
    for m in range(3):
        for i in range(len(predys[0])):
            dp_dt=np.gradient(predys[m][i].flatten())
            dy_dt=np.gradient(testy[i].flatten())
            P=predys[m][i].flatten()
            T=testy[i].flatten()
            div=P.shape[0]
            u=np.sum(np.abs(P-T)/div)
            v=np.sum(np.abs(dy_dt-dp_dt)/div)
            M[m,i]=u
            Md[m,i]=v
    sb.boxplot(data=M.T)
    plt.show()
    sb.boxplot(data=Md.T)
    plt.show()
            
if __name__=='__main__':
    incrange=50
    hidden_nodes=32
    base='/mnt/D2/Chaos/mg/lng/stable/L50'
    ident='A'

    spacing=100
    data=np.load('{}/data{}.npy'.format(base,ident))
    i=5
    mname='ELM'
    predy=np.load('{}/{}_{}_H{}.npy'.format(base,mname,i,hidden_nodes))
    #predy=np.transpose(predy,[1,2,0])

    train,test=bisect_set(data,i,incrange,spacing)
    print(predy.shape)
    testy=test[:,100*i:]
    print (predy.shape,testy.shape)
    gradplot(predy,testy,mname,i)
    
    P=(np.load('{}/ELM_{}_H{}.npy'.format(base,i,hidden_nodes)),np.load('{}/LSTM_{}_H{}.npy'.format(base,i,hidden_nodes)),np.load('{}/GRU_{}_H{}.npy'.format(base,i,hidden_nodes)))

    #box_plot(P,testy,mname,i)
