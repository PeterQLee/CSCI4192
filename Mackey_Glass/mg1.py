import numpy as np
from models import LSTM, GRU, Elman, reset_graph
import sys
import os

def bisect_set(data,ahead,incrange,spacing=100):
    '''
    data
    ahead (i.e. how far to forcast
    incrange (how long a series should be ahead
    Spacing, increment
    '''
    
    #10% test 90% train
    data_=data[20*spacing:]
    div=len(data_)//10*9
    
    train=data_[:div]
    test=data_[div:]
    trainx=np.zeros((div//(incrange*spacing),(incrange)*100))
    testx=np.zeros((len(data_)//10//(incrange*spacing),(incrange)*100))
    c=0

    print(div,div//(incrange*spacing))
    for i in range(0,div-incrange*spacing,incrange*spacing):
        trainx[c]=data[i:i+incrange*spacing]
        c+=1
    c=0
    for i in range(div,len(data_)-incrange*spacing,incrange*spacing):
        testx[c]=data[i:i+incrange*spacing]
        c+=1

    return trainx,testx

def one_a(incrange,spacing=100):
    for n_hidden in [64]:#[32,64,96,128]:
        base='/mnt/D2/Chaos/mg/lng/stable/L50/'
        os.makedirs(base,exist_ok=True)
        ident='A'
        data=np.load('{}/data{}.npy'.format(base,ident))
        for i in [1]:

            train,test=bisect_set(data,i,incrange,spacing)
            #np.random.seed(0)
            #inds=np.arange(len(train))
            #np.random.shuffle(inds)
            if not os.path.exists('{}/GRU_{}_H{}r.npy'.format(base,i,n_hidden)):
                print('GRU',i,n_hidden)
                reset_graph()
                model=GRU(1,1)
                err=model.train(train[:,:-i*spacing],train[:,i*spacing:])
                predy=model.predict(test[:,:-i*spacing])
                np.save('{}/GRU_{}_H{}r.npy'.format(base,i,n_hidden),predy)
                np.save('{}/GRU_{}err_H{}r.npy'.format(base,i,n_hidden),err)
                
            if not os.path.exists('{}/ELM_{}_H{}r.npy'.format(base,i,n_hidden)):
                print('ELU',i,n_hidden)
                reset_graph()
                model=Elman(1,1) #Need to fix Elman transpose predictions
                err=model.train(train[:,:-i*spacing],train[:,i*spacing:])
                predy=model.predict(test[:,:-i*spacing])
                np.save('{}/ELM_{}_H{}r.npy'.format(base,i,n_hidden),predy)
                np.save('{}/ELM_{}err_H{}r.npy'.format(base,i,n_hidden),err)

            if not os.path.exists('{}/LSTM_{}_H{}r.npy'.format(base,i,n_hidden)):
                print('LSTM',i,n_hidden)
                reset_graph()
                model=LSTM(1,1)
                err=model.train(train[:,:-i*spacing],train[:,i*spacing:])
                predy=model.predict(test[:,:-i*spacing])
                np.save('{}/LSTM_{}_H{}r.npy'.format(base,i,n_hidden),predy)
                np.save('{}/LSTM_{}err_H{}r.npy'.format(base,i,n_hidden),err)


                 
def one_revised(spacing=100):
    base='/mnt/D2/Chaos/mg/lng/stable2/'
    os.makedirs(base,exist_ok=True)
    ident='A'
    data=np.load('{}/data{}.npy'.format(base,ident))
    train=data[:len(data)//2]
    test=data[len(data)//2:]
    train=train[::spacing]
    test=test[::spacing]
    train=train.reshape(1,-1)
    test=test.reshape(1,-1)


    for n_hidden in [128,96,64,32]:#[32,64,96,128]:        
        for i in [1,5,10]:
        
            if not os.path.exists('{}/GRU_{}_H{}.npy'.format(base,i,n_hidden)):
                print('GRU',i,n_hidden)
                reset_graph()
                model=GRU(1,1,n_hidden)
                err=model.train(train[:,:-i],train[:,i:],ep=200)
                predy=model.predict(test[:,:-i])
                np.save('{}/GRU_{}_H{}.npy'.format(base,i,n_hidden),predy)
                np.save('{}/GRU_{}err_H{}.npy'.format(base,i,n_hidden),err)

            if not os.path.exists('{}/ELM_{}_H{}.npy'.format(base,i,n_hidden)):
                print('ELU',i,n_hidden)
                reset_graph()
                model=Elman(1,1,n_hidden) #Need to fix Elman transpose predictions
                err=model.train(train[:,:-i],train[:,i:],ep=200)
                predy=model.predict(test[:,:-i])
                np.save('{}/ELM_{}_H{}.npy'.format(base,i,n_hidden),predy)
                np.save('{}/ELM_{}err_H{}.npy'.format(base,i,n_hidden),err)

            if not os.path.exists('{}/LSTM_{}_H{}.npy'.format(base,i,n_hidden)):
                print('LSTM',i,n_hidden)
                reset_graph()
                model=LSTM(1,1,n_hidden)
                err=model.train(train[:,:-i],train[:,i:],ep=200)
                predy=model.predict(test[:,:-i])
                np.save('{}/LSTM_{}_H{}.npy'.format(base,i,n_hidden),predy)
                np.save('{}/LSTM_{}err_H{}.npy'.format(base,i,n_hidden),err)



if __name__=='__main__':
    
    #one_a(50)
    one_revised()
