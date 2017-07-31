import numpy as np
from mg1 import bisect_set
from models import LSTM, GRU, Elman,MLP, reset_graph
import sys
import os

def mlp_split(data,ahead):
    x0=1.2
    X=np.zeros((data.shape[0]*(data.shape[1]-ahead),4))
    X[:,:]=x0
    Y=np.zeros((data.shape[0]*(data.shape[1]-ahead),1))
    for k in range(data.shape[0]):
        N=k*(data.shape[1]-ahead)
        for i in range(data.shape[1]-ahead):
            for j in range(0,4,1):
                if i+6*j>=data.shape[1]-ahead:
                    break
                X[k+i+6*j,j]=data[k,i]
            Y[k+i,0]=data[k,i+ahead]

    inds=np.arange(data.shape[0]*(data.shape[1]-ahead))
    np.random.shuffle(inds)
    return X[inds],Y[inds]

def two(spacing=100):

    base='/mnt/D2/Chaos/mg/lng/rho/'
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
        
    for v in range(10):
        #v is the testset
        test=test_data[v]
        train=np.concatenate([train_data[p] for p in range(10) if p!=v],axis=0)
        inds=np.arange(train.shape[0])
        np.random.shuffle(inds)
        train=train[inds]
        for n_hidden in [128,96,64,32]:#[32,64,96,128]:        
            for i in [1,5,10,30]:
                n_h=str(n_hidden)+chr(ord('a')+v)
                #n_h=str(n_hidden)
                
                # if not os.path.exists('{}/MLP_{}_H{}.npy'.format(base,i,n_h)):
                #     print('MLP',i,n_h)
                #     reset_graph()
                #     model=MLP(4,1,n_hidden)
                #     X,Y=mlp_split(train,i)
                #     print(train.shape,X.shape,Y.shape)
                #     err=model.train(X,Y,ep=50)
                #     X,_=mlp_split(test,i)
                #     predy=model.predict(X)
                #     np.save('{}/MLP_{}_H{}.npy'.format(base,i,n_h),predy)
                #     np.save('{}/MLP_{}err_H{}.npy'.format(base,i,n_h),err)

                if not os.path.exists('{}/GRU_{}_H{}.npy'.format(base,i,n_h)):
                    print('GRU',i,n_h)
                    reset_graph()
                    model=GRU(1,1,n_hidden)
                    err=model.train(train[:,:-i],train[:,i:],ep=50)
                    predy=model.predict(test[:,:-i])
                    np.save('{}/GRU_{}_H{}.npy'.format(base,i,n_h),predy)
                    np.save('{}/GRU_{}err_H{}.npy'.format(base,i,n_h),err)

                if not os.path.exists('{}/ELM_{}_H{}.npy'.format(base,i,n_h)):
                    print('ELU',i,n_h)
                    reset_graph()
                    model=Elman(1,1,n_hidden) #Need to fix Elman transpose predictions
                    err=model.train(train[:,:-i],train[:,i:],ep=50)
                    predy=model.predict(test[:,:-i])
                    np.save('{}/ELM_{}_H{}.npy'.format(base,i,n_h),predy)
                    np.save('{}/ELM_{}err_H{}.npy'.format(base,i,n_h),err)

                if not os.path.exists('{}/LSTM_{}_H{}.npy'.format(base,i,n_h)):
                    print('LSTM',i,n_h)
                    reset_graph()
                    model=LSTM(1,1,n_hidden)
                    err=model.train(train[:,:-i],train[:,i:],ep=50)
                    predy=model.predict(test[:,:-i])
                    np.save('{}/LSTM_{}_H{}.npy'.format(base,i,n_h),predy)
                    np.save('{}/LSTM_{}err_H{}.npy'.format(base,i,n_h),err)


if __name__=='__main__':
    
    two()
