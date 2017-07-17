import tensorflow as tf
import numpy as np

class Network:
    def reset(self):
        self.sess.run(tf.global_variables_initializer())
    def train(self,X_T,Y_T,ep=1):
        #self.sess.run(tf.global_variables_initializer())
        X,Y_,train_step,error=[self.retrieve[i] for i in ['X','Y_','train_step','error']]
        
        all_acc=np.zeros(X_T.shape[0]*ep)
        for e in range(ep):
            for i in range(len(X_T)):
                x=X_T[i]
                y=Y_T[i]
                x=x.reshape(1,-1,self.n_in)
                y=y.reshape(-1,1)
                _,acc=self.sess.run([train_step,error],feed_dict={X:x,Y_:y})
                all_acc[e*len(X_T)+i]=acc

        return all_acc

    # def predict(self,Xt):
    #     output=self.retrieve['output']
    #     X=self.retrieve['X']
    #     out,=self.sess.run([output],feed_dict={X:Xt.reshape(Xt.shape[0],Xt.shape[1],self.n_in)})

    #     return out

    def predict(self,Xt):
        output=self.retrieve['output']
        X=self.retrieve['X']
        O=np.zeros(Xt.shape)
        for i in range(Xt.shape[0]):
            out,=self.sess.run([output],feed_dict={X:Xt.reshape(1,Xt.shape[1],self.n_in)})
            O[i]=out

        return O


class LSTM (Network):
    def __init__(self,n_in,n_out):
        self.sess=tf.Session()
        self.n_in=n_in
        X=tf.placeholder(tf.float32,shape=[None,None,n_in])
        Y_=tf.placeholder(tf.float32,shape=[None,n_out])

        lstm_size=64
        lstm=tf.contrib.rnn.BasicLSTMCell(lstm_size)
        lstm1=tf.contrib.rnn.BasicLSTMCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        #V=tf.reshape(tf.tile(V,[tf.shape(rnout)[0],1]),[-1,lstm_size,1])
        output=tf.matmul(rnout[0],V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))
        train_step=tf.train.AdamOptimizer(5e-4).minimize(error)

        rlist=[output,error,train_step,X,Y_]
        rname=['output','error','train_step','X','Y_']

        self.retrieve=dict(zip(rname,rlist))

class GRU (Network):
    def __init__(self,n_in,n_out):
        self.sess=tf.Session()
        self.n_in=n_in
        X=tf.placeholder(tf.float32,shape=[None,None,n_in])
        Y_=tf.placeholder(tf.float32,shape=[None,n_out])

        lstm_size=32
        lstm=tf.contrib.rnn.GRUCell(lstm_size)
        lstm1=tf.contrib.rnn.GRUCell(lstm_size)
        stacklstm=tf.contrib.rnn.MultiRNNCell([lstm,lstm1],state_is_tuple=True)
        rnout,state=tf.nn.dynamic_rnn(stacklstm,X,dtype=tf.float32)
        Bv=tf.Variable(tf.constant(0.,shape=[1]),name='Bv')
        V=tf.get_variable(name='V',shape=[lstm_size,1],initializer=tf.contrib.layers.xavier_initializer())
        output=tf.matmul(rnout[0],V)+Bv #Linear unit
        error=tf.reduce_mean(tf.square(output-Y_))
        train_step=tf.train.AdamOptimizer(1e-3).minimize(error)

        rlist=[output,error,train_step,X,Y_]
        rname=['output','error','train_step','X','Y_']

        self.retrieve=dict(zip(rname,rlist))



