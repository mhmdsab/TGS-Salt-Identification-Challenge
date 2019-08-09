class Early_Stopping:
    
    def __init__(self, sess, saver, epochs_to_wait):
        self.sess = sess
        self.saver = saver
        self.epochs_to_wait = epochs_to_wait
        self.test_loss_list = []
        self.counter = 0
        
    def add_loss(self):
        self.test_loss_list.append(self.test_loss)
        
    def save_best_model(self,test_loss):
        self.test_loss = test_loss
        self.add_loss()
        
        if min(self.test_loss_list) == self.test_loss_list[-1]:
            self.best_loss = self.test_loss
            self.saver.save(self.sess, 'D:/MSabLib/U_net_res_blocks_101/checkpoints/')
            self.counter = 0
            
        else:
            print('loss did not improve since test loss was {}'.format(self.best_loss))
            self.counter += 1

        
        
        
        
