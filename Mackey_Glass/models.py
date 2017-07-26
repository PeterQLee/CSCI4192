import tensorflow as tf
import numpy as np
from queuefetcher import QueueFetcher
def reset_graph():
    tf.reset_default_graph()

class Network:
    def __init__(self,n_in,n_out):
        self.sess=tf.Session()
        self.input_Q=tf.placeholder(tf.float32,shape=[None,None,n_in])
        self.label_Q=tf.placeholder(tf.float32,shape=[None,None,n_out])
        self.learning_Q=tf.placeholder(tf.float32,shape=[None])
        self.Q_fetch=QueueFetcher(['float32','float32','float32'],[self.input_Q,self.label_Q,self.learning_Q],self.sess)
        self.Q_fetch.start()
        
    def reset(self):
        self.sess.run(tf.global_variables_initializer())
        
    def train(self,X_T,Y_T,ep=1):
        batchsize=20
        self.sess.run(tf.global_variables_initializer())
        train_step,error=[self.retrieve[i] for i in ['train_step','error']]
        all_acc=[]
        #Prep learning rates
        # learnrates=[1e-3 for i in range(X_T.shape[0]//2)] + [1e-4 for i in range(X_T.shape[0]//3)] + [1e-5 for i in range(X_T.shape[0]-(X_T.shape[0]//2+X_T.shape[0]//3))]
        # self.Q_fetch.enqueue_data((X_T.reshape(X_T.shape[0],X_T.shape[1],1),Y_T.reshape(Y_T.shape[0],Y_T.shape[1],1),learnrates),[],batchsize=batchsize,shuffle=False)

        # for e in range(ep):
        #     self.Q_fetch.training_event.wait()

        #     self.Q_fetch.start_epoch()
        #     for i in range(0,len(X_T),batchsize):
        #         _,acc=self.sess.run([train_step,error])
        #         print('donewait')
        #         all_acc.append(acc)

        # #Do smaller portion with a batchsize of 1
        # X_tt=X_T[:100]
        # Y_tt=Y_T[:100]
        X_tt=X_T
        Y_tt=Y_T
        batchsize=1
        #Prep learning rates
        learnrates=[1e-3 for i in range(X_tt.shape[0])] 
        #learnrates=[1e-3 for i in range(X_tt.shape[0]//2)] + [1e-4 for i in range(X_tt.shape[0]//3)] + [1e-5 for i in range(X_tt.shape[0]-(X_tt.shape[0]//2+X_tt.shape[0]//3))]
        self.Q_fetch.enqueue_data((X_tt.reshape(X_tt.shape[0],X_tt.shape[1],1),Y_tt.reshape(Y_tt.shape[0],Y_tt.shape[1],1),learnrates),[],batchsize=batchsize,shuffle=False)

        for e in range(ep):
            self.Q_fetch.training_event.wait()

            self.Q_fetch.start_epoch()
            for i in range(0,len(X_tt),batchsize):
                _,acc=self.sess.run([train_step,error])
                print('donewait')
                #all_acc[e*len(X_tt)+i]=acc
                all_acc.append(acc)

        self.Q_fetch.stop()
        return np.array(all_acc)

    def predict(self,Xt):
        output=self.retrieve['output']
        X=self.retrieve['X']
        out,=self.sess.run([output],feed_dict={X:Xt.reshape(Xt.shape[0],Xt.shape[1],self.n_in)})

        return out

    # def predict(self,Xt):
    #     output=self.retrieve['output']
    #     X=self.retrieve['X']
    #     O=np.zeros(Xt.shape)
    #     for i in range(Xt.shape[0]):
    #         out,=self.sess.run([output],feed_dict={X:Xt.reshape(1,Xt.shape[1],self.n_in)})
    #         O[i]=out

    #     return O


class BLSTM (Network):
    def __init__(self,n_in,n_out,lstm_size=64):
        super().__init__(n_in,n_out)
        self.n_in=n_in
        X,Y_,learnrate=self.Q_fetch.Q.dequeue()
        X.set_shape([None, None, n_in])
        Y_.set_shape([ None, n_out])

        learnrate=learnrate[0]
        #X=tf.placeholder(tf.float32,shape=[None,None,n_in])
        #Y_=tf.placeholder(tf.float32,shape=[None,n_out])

        lstm=tf.contrib.rnn.BasicLSTMCell(lstm_size)
        lstm1=tf.contrib.rnn.BasicLSTMCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        V=tf.reshape(tf.tile(V,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
        output=tf.matmul(rnout,V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))
        train_step=tf.train.AdamOptimizer(learnrate).minimize(error)

        rlist=[output,error,train_step,X,Y_,self.input_Q,self.label_Q]
        rname=['output','error','train_step','X','Y_','input_Q','label_Q']

        self.retrieve=dict(zip(rname,rlist))

class LSTM (Network):
    def __init__(self,n_in,n_out,lstm_size=64):
        super().__init__(n_in,n_out)
        self.n_in=n_in
        X,Y_,learnrate=self.Q_fetch.Q.dequeue()
        X.set_shape([None, None, n_in])
        Y_.set_shape([None, None, n_out])
        learnrate=learnrate[0]
        #X=tf.placeholder(tf.float32,shape=[None,None,n_in])
        #Y_=tf.placeholder(tf.float32,shape=[None,n_out])

        lstm=tf.contrib.rnn.LSTMCell(lstm_size)
        lstm1=tf.contrib.rnn.LSTMCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        V=tf.reshape(tf.tile(V,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
        output=tf.matmul(rnout,V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))

        # lambda_l2_reg=0.1
        # l2 = lambda_l2_reg * sum(
        #         tf.nn.l2_loss(tf_var)
        #             for tf_var in tf.trainable_variables()
        #     if not ("noreg" in tf_var.name or "Bias" in tf_var.name or 'B' in tf_var.name)
        # )
        
        # E=error+l2
        E=error
        train_step=tf.train.AdamOptimizer(learnrate).minimize(E)

        rlist=[output,error,train_step,X,Y_,self.input_Q,self.label_Q]
        rname=['output','error','train_step','X','Y_','input_Q','label_Q']

        self.retrieve=dict(zip(rname,rlist))

class GRU (Network):
    def __init__(self,n_in,n_out,lstm_size):
        super().__init__(n_in,n_out)
        self.n_in=n_in
        X,Y_,learnrate=self.Q_fetch.Q.dequeue()

        X.set_shape([None, None, n_in])
        Y_.set_shape([None, None, n_out])
        learnrate=learnrate[0]

        lstm=tf.contrib.rnn.GRUCell(lstm_size)
        lstm1=tf.contrib.rnn.GRUCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        V=tf.reshape(tf.tile(V,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
        output=tf.matmul(rnout,V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))
        # lambda_l2_reg=0.1
        # l2 = lambda_l2_reg * sum(
        #         tf.nn.l2_loss(tf_var)
        #             for tf_var in tf.trainable_variables()
        #     if not ("noreg" in tf_var.name or "Bias" in tf_var.name or 'B' in tf_var.name)
        # )
        
        # E=error+l2
        E=error

        train_step=tf.train.AdamOptimizer(learnrate).minimize(E)

        rlist=[output,error,train_step,X,Y_,self.input_Q,self.label_Q]
        rname=['output','error','train_step','X','Y_','input_Q','label_Q']

        self.retrieve=dict(zip(rname,rlist))



class Elman():
    def __init__(self,n_in,n_out,n_hidden=64):
        #super().__init__(n_in,n_out)

        self.n_in=n_in
        
        
        X=tf.placeholder(tf.float32,shape=[None, None, n_in])
        Y_=tf.placeholder(tf.float32,shape=[None, None, n_out])

        shapes={'X':[None,n_in],
                'Y_':[n_out],
                'W1':[n_in,n_hidden],
                'RW1':[n_hidden,n_hidden],
                'B1':[n_hidden],
                'W2':[n_hidden,n_hidden],
                'RW2':[n_hidden,n_hidden],
                'B2':[n_hidden],
                'W3':[n_hidden,n_out],
                'B3':[n_out],
                'H1':[1,n_hidden],
                'H2':[1,n_hidden]
                }

        learnrate=tf.placeholder(tf.float32)

        W1=tf.get_variable('W1',shapes['W1'],initializer=tf.contrib.layers.xavier_initializer())
        RW1=tf.get_variable('RW1',shapes['RW1'],initializer=tf.contrib.layers.xavier_initializer())
        B1=tf.get_variable('B1',initializer=np.zeros(shapes['B1'],dtype=np.float32))
        W2=tf.get_variable('W2',shapes['W2'],initializer=tf.contrib.layers.xavier_initializer())
        RW2=tf.get_variable('RW2',shapes['RW2'],initializer=tf.contrib.layers.xavier_initializer())
        B2=tf.get_variable('B2',initializer=np.zeros(shapes['B2'],dtype=np.float32))

        #X shape (batch, tseries, n_in)
        fillshape=tf.concat([tf.shape(X)[0:1],[n_hidden]],axis=0)
        filler=tf.fill(fillshape,0.)
        self.internal_params={'W1':W1,'RW1':RW1,'B1':B1,'W2':W2,'RW2':RW2,'B2':B2}

        H1,H2=tf.scan(self.push_recurrence,tf.transpose(X,[1,0,2]),initializer=(filler,filler))

        W3=tf.get_variable('W3',shapes['W3'],initializer=tf.contrib.layers.xavier_initializer())
        B3=tf.get_variable('B3',initializer=np.zeros(shapes['B3'],dtype=np.float32))
        W3=tf.reshape(tf.tile(W3,[tf.shape(H2)[0],1]),[-1,n_hidden,1])
        y=tf.transpose(tf.matmul(H2,W3)+B3,[1,0,2])

        error=tf.reduce_mean(tf.square(y-Y_))
        # lambda_l2_reg=0.1
        # l2 = lambda_l2_reg * sum(
        #         tf.nn.l2_loss(tf_var)
        #             for tf_var in tf.trainable_variables()
        #     if not ("noreg" in tf_var.name or "Bias" in tf_var.name or 'B' in tf_var.name)
        # )
        
        # E=error+l2

        E=error
        train_step=tf.train.AdamOptimizer(learnrate).minimize(E)
        

        self.retrieve={'train_step':train_step,'error':error,'X':X,'Y_':Y_,'learnrate':learnrate,'y':y,'W3':W3,'H2':H2}
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print('setup')
        
    def push_recurrence(self,states,inp):
        #with tf.variable_scope("RNN"):
        rH1,rH2=states
        #lW1=self.interal_params
        #lW2=self.interal_params
        H1=tf.nn.tanh(tf.matmul(inp,self.internal_params['W1'])+tf.matmul(rH1,self.internal_params['RW1'])+self.internal_params['B1'])
        H2=tf.nn.tanh(tf.matmul(H1,self.internal_params['W2'])+tf.matmul(rH2,self.internal_params['RW2'])+self.internal_params['B2'])
        return H1,H2

    def train(self,X_T,Y_T,ep=1):
        batchsize=10
        self.sess.run(tf.global_variables_initializer())
        train_step,error=[self.retrieve[i] for i in ['train_step','error']]
        all_acc=[]
        #Prep learning rates
        # learnrates=[1e-3 for i in range(X_T.shape[0]//2)] + [1e-4 for i in range(X_T.shape[0]//3)] + [1e-5 for i in range(X_T.shape[0]-(X_T.shape[0]//2+X_T.shape[0]//3))]

        # sdat=(X_T.reshape(X_T.shape[0],X_T.shape[1],1),Y_T.reshape(Y_T.shape[0],Y_T.shape[1],1),learnrates)
        # print('S',sdat[0].shape,sdat[1].shape)
        # stens=[self.retrieve[i] for i in ['X','Y_','learnrate']]
        
        # for e in range(ep):
        #     for i in range(0,len(X_T),batchsize):
        #         #v=[np.expand_dims(sdat[j][i],axis=0) for j in range(3)]
        #         v=[sdat[j][i:i+batchsize] for j in range(3)]

        #         v[2]=v[2][0]
        #         _,acc=self.sess.run([train_step,error],feed_dict=dict(zip(stens,v)))
        #         #all_acc[e*len(X_T)+i]=acc
        #         all_acc.append(acc)
        #         print(acc)

        # X_tt=X_T[:100]
        # Y_tt=Y_T[:100]
        
        X_tt=X_T
        Y_tt=Y_T
        batchsize=1

        learnrates=[1e-3 for i in range(X_tt.shape[0])] 
        #learnrates=[1e-3 for i in range(X_tt.shape[0]//2)] + [1e-4 for i in range(X_tt.shape[0]//3)] + [1e-5 for i in range(X_tt.shape[0]-(X_tt.shape[0]//2+X_tt.shape[0]//3))]
        sdat=(X_tt.reshape(X_tt.shape[0],X_tt.shape[1],1),Y_tt.reshape(Y_tt.shape[0],Y_tt.shape[1],1),learnrates)
    
        print('S',sdat[0].shape,sdat[1].shape)
        stens=[self.retrieve[i] for i in ['X','Y_','learnrate']]

        for e in range(ep):
            for i in range(0,len(X_tt),batchsize):
                v=[sdat[j][i:i+batchsize] for j in range(3)]
                v[2]=v[2][0]
                _,acc=self.sess.run([train_step,error],feed_dict=dict(zip(stens,v)))
                #all_acc[e*len(X_tt)+i]=acc
                all_acc.append(acc)
                print(acc)

        return np.array(all_acc)

    
    def predict(self,Xt):
        output=self.retrieve['y']
        X=self.retrieve['X']
        out,=self.sess.run([output],feed_dict={X:Xt.reshape(Xt.shape[0],Xt.shape[1],self.n_in)})

        return out

    def reset(self):
        self.sess.run(tf.global_variables_initializer())

class MLP:
    def __init__(self,n_in,n_out,n_hidden):
        X=tf.placeholder(tf.float32,shape=[None, None, n_in])
        Y_=tf.placeholder(tf.float32,shape=[None, None, n_out])

        shapes={'X':[None,n_in],
                'Y_':[n_out],
                'W1':[n_in,n_hidden],
                'RW1':[n_hidden,n_hidden],
                'B1':[n_hidden],
                'W2':[n_hidden,n_hidden],
                'RW2':[n_hidden,n_hidden],
                'B2':[n_hidden],
                'W3':[n_hidden,n_out],
                'B3':[n_out],
                'H1':[1,n_hidden],
                'H2':[1,n_hidden]
                }
