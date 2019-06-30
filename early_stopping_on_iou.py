class Early_Stopping:
    
    def __init__(self, sess, saver, epochs_to_wait):
        self.sess = sess
        self.saver = saver
        self.epochs_to_wait = epochs_to_wait
        self.iou_list = []
        self.counter = 0
        
    def add_iou(self):
        self.iou_list.append(self.iou)
        
    def save_best_model(self,iou):
        self.iou = iou
        self.add_iou()
        
        if max(self.iou_list) == self.iou_list[-1]:
            self.best_iou = self.iou
            self.saver.save(self.sess, 'D:/MSabLib/U_net_res_blocks_101/checkpoints/')
            self.counter = 0
            
        else:
            print('iou did not improve since test iou was {}'.format(self.best_iou))
            self.counter += 1

        
        
        
        