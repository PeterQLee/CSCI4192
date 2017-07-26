import tensorflow as tf
import numpy as np
import threading

class QueueFetcher(threading.Thread):
    def __init__(self,dtypes,tensors,sess,capacity=2000):
        '''
        Initializes parameters

        tensors :list of tensors
        '''
        super(QueueFetcher,self).__init__()
        
        self.tensors=tensors
        self.Q=tf.FIFOQueue(capacity=capacity,dtypes=dtypes)
        self.enqueue_op=self.Q.enqueue(tensors)
        #In the future, it may be good to include RandomshuffleQueue.
        #However, this would require some graph manipulation I don't know how to do
        
        self.stop_flag=False
        self.epoch_event=threading.Event()
        self.training_event=threading.Event()
        self.training_event.clear()
        self.epoch_event.clear()
        self.testing_curve=False
        self.TEST_THRESHOLD=500

        self.qr=tf.train.QueueRunner(self.Q,[self.enqueue_op])
        self.coord=tf.train.Coordinator()
        self.threads=tf.train.start_queue_runners(coord=self.coord,sess=sess)
        tf.train.add_queue_runner(self.qr)
        self.sess=sess

    def _shuffle(self,data):
        '''
        Shuffles data in place
        '''
        state=np.random.get_state()
        for k in range(len(data)):
            np.random.shuffle(data[k])
            np.random.set_state(state)
            
    def enqueue_data(self,data,constants,batchsize=50,shuffle=False):
        '''
        Places a full chunk of data into the queue
        For incremental placement, use  enqueue_datainc
        '''
        #TODO:
        #Transfer shuffle into a more sophisticated data augmentation class
        self.shuffle=shuffle        
        self.data=data
        self.batchsize=batchsize
        self.constants=constants
        self.batches_per_epoch=np.ceil(len(data)/batchsize)
        if self.shuffle:
            self._shuffle(self.data)

    def enqueue_testdata(self,data,constants,thresh):
        '''
        '''
        self.TEST_THRESHOLD=thresh
        self.testing_curve=True
        
        self.testdata=data
        self.testconstants=constants

        
    # def enqueue_datainc(self,data,shape=shape):
    #     '''
    #     concatenates
    #     '''

    #     #TODO: implement flag locks
    #     self.data=np.concatenate((self.data,data))
        
    def stop(self):
        '''
        Stops thread
        '''

        self.stop_flag=True
        self.epoch_event.set()
        
        ## stop queue runners, close queue
        self.coord.request_stop()
        print('Stopping Queuerunner')
        self.coord.join()
        self.Q.close()
        
    def start_epoch(self):
        '''
        Tells the queue fetcher to obtain another epoch
        '''
        self.epoch_event.set()
        
    def run(self):
        '''
        Thread run
        '''
        self.training_event.set()
        self.epoch_event.wait()
        
        i=0
        while not self.stop_flag:
            c=0
            #assert i==0 or i==(len(self.data[0])//self.batchsize)*self.batchsize
            #Run through a full epoch
            
            self.training_event.clear()
            
            for i in range(0,len(self.data[0]),self.batchsize):
                #Test
                D=[k[i:i+self.batchsize] for k in self.data]+[self.constants]
                self.sess.run(self.enqueue_op,feed_dict=dict(zip(self.tensors,D)))
                if self.testing_curve and c%self.TEST_THRESHOLD==0:

                    D=list(self.data)+[self.testconstants]
                    self.sess.run(self.enqueue_op,feed_dict=dict(zip(self.tensors,D)))

                c+=1
                
            D=[k[i:] for k in self.data]+[self.constants]
            self.sess.run(self.enqueue_op,feed_dict=dict(zip(self.tensors,D)))
            self.epoch_event.clear()
            if self.shuffle:
                self._shuffle(self.data)
            self.training_event.set()
            self.epoch_event.wait()
    
