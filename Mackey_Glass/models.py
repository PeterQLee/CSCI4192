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
        
        self.Q_fetch=QueueFetcher(['float32','float32'],[self.input_Q,self.label_Q],self.sess)
        self.Q_fetch.start()
    def reset(self):
        self.sess.run(tf.global_variables_initializer())
    def train(self,X_T,Y_T,ep=1):
        self.sess.run(tf.global_variables_initializer())
        train_step,error=[self.retrieve[i] for i in ['train_step','error']]
        print(X_T.shape)
        #Prep learning rates
        self.Q_fetch.enqueue_data((X_T.reshape(X_T.shape[0],X_T.shape[1],1),Y_T.reshape(Y_T.shape[0],Y_T.shape[1],1)),[],batchsize=1,shuffle=False)
        all_acc=np.zeros(X_T.shape[0]*ep)
        for e in range(ep):
            self.Q_fetch.training_event.wait()
            self.Q_fetch.start_epoch()
            for i in range(len(X_T)):
                #x=X_T[i]
                #y=Y_T[i]
                #x=x.reshape(1,-1,self.n_in)
                #y=y.reshape(-1,1)
                
                _,acc=self.sess.run([train_step,error])
                all_acc[e*len(X_T)+i]=acc

        self.Q_fetch.stop()
        return all_acc

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
        X,Y_=self.Q_fetch.Q.dequeue()
        X.set_shape([None, None, n_in])
        Y_.set_shape([ None, n_out])

        #X=tf.placeholder(tf.float32,shape=[None,None,n_in])
        #Y_=tf.placeholder(tf.float32,shape=[None,n_out])

        lstm_size=64
        lstm=tf.contrib.rnn.BasicLSTMCell(lstm_size)
        lstm1=tf.contrib.rnn.BasicLSTMCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        V=tf.reshape(tf.tile(V,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
        output=tf.matmul(rnout,V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))
        train_step=tf.train.AdamOptimizer(5e-4).minimize(error)

        rlist=[output,error,train_step,X,Y_,self.input_Q,self.label_Q]
        rname=['output','error','train_step','X','Y_','input_Q','label_Q']

        self.retrieve=dict(zip(rname,rlist))

class LSTM (Network):
    def __init__(self,n_in,n_out,lstm_size=64):
        super().__init__(n_in,n_out)
        self.n_in=n_in
        X,Y_=self.Q_fetch.Q.dequeue()
        X.set_shape([None, None, n_in])
        Y_.set_shape([ None, n_out])

        #X=tf.placeholder(tf.float32,shape=[None,None,n_in])
        #Y_=tf.placeholder(tf.float32,shape=[None,n_out])

        lstm_size=64
        lstm=tf.contrib.rnn.LSTMCell(lstm_size)
        lstm1=tf.contrib.rnn.LSTMCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        V=tf.reshape(tf.tile(V,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
        output=tf.matmul(rnout,V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))
        train_step=tf.train.AdamOptimizer(5e-4).minimize(error)

        rlist=[output,error,train_step,X,Y_,self.input_Q,self.label_Q]
        rname=['output','error','train_step','X','Y_','input_Q','label_Q']

        self.retrieve=dict(zip(rname,rlist))

class GRU (Network):
    def __init__(self,n_in,n_out):
        super().__init__(n_in,n_out)
        self.n_in=n_in
        X,Y_=self.Q_fetch.Q.dequeue()
        X.set_shape([None, None, n_in])
        Y_.set_shape([ None, n_out])
        lstm_size=64
        lstm=tf.contrib.rnn.GRUCell(lstm_size)
        lstm1=tf.contrib.rnn.GRUCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        V=tf.reshape(tf.tile(V,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
        output=tf.matmul(rnout,V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))
        train_step=tf.train.AdamOptimizer(5e-4).minimize(error)

        rlist=[output,error,train_step,X,Y_,self.input_Q,self.label_Q]
        rname=['output','error','train_step','X','Y_','input_Q','label_Q']

        self.retrieve=dict(zip(rname,rlist))



class Elman(Network):
    def __init__(self,n_in,n_out):
        super().__init__(n_in,n_out)
        
        self.n_in=n_in
        with tf.variable_scope("RNN"):
            X,Y_=self.Q_fetch.Q.dequeue()
            X.set_shape([None, None, n_in])
            Y_.set_shape([ None, n_out])

            W1=tf.get_variable('W1',shapes['W1'],initializer=tf.contrib.layers.xavier_initializer())
            RW1=tf.get_variable('RW1',shapes['RW1'],initializer=tf.contrib.layers.xavier_initializer())
            B1=tf.get_variable('B1',initializer=np.zeros(shapes['B1'],dtype=np.float32))
            W2=tf.get_variable('W2',shapes['W2'],initializer=tf.contrib.layers.xavier_initializer())
            RW2=tf.get_variable('RW2',shapes['RW2'],initializer=tf.contrib.layers.xavier_initializer())
            B2=tf.get_variable('B2',initializer=np.zeros(shapes['B2'],dtype=np.float32))
        with tf.variable_scope('RNN',reuse=True):
            
            H1,H2=tf.scan(self.recurrent_hidden,X,parallel_iterations=50,initializer=(np.random.uniform(-1,1,shapes['H1']).astype(np.float32),np.random.uniform(-1,1,shapes['H2']).astype(np.float32)))
                        
        with tf.variable_scope('RNN'):
            W3=tf.get_variable('W3',shapes['W3'],initializer=tf.contrib.layers.xavier_initializer())
            B3=tf.get_variable('B3',initializer=np.zeros(shapes['B3'],dtype=np.float32))
            W3=tf.reshape(tf.tile(W3,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
            y=tf.matmul(rnout,W3)+B3 #Next try linear combination of values
            #yp=tf.nn.softmax(y)

            #error=tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=y,name="error")
            error=tf.reduce_mean(tf.square(y-Y_))
            train=tf.train.AdamOptimizer(5e-4).minimize(error)

            self.retrieve={'train':train,'error':error,'X':X,'yp':yp,'Y_':Y_,'keep_chance':keep_chance}
            self.sess=tf.Session()
            self.sess.run(tf.global_variables_initializer())
        
    def push_recurrence(self,states,inp):
        #with tf.variable_scope("RNN"):
        rH1,rH2=states
        H1=tf.matmul(tf.reshape(inp,[-1,1]),tf.get_variable('W1'))+tf.matmul(rH1,tf.get_variable('RW1'))+tf.get_variable('B1')
        H2=tf.matmul(tf.nn.dropout(H1,self.keep_chance),tf.get_variable('W2'))+tf.matmul(rH2,tf.get_variable('RW2'))+tf.get_variable('B2')
        return H1,H2
