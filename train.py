import cv2
import tensorflow as tf
import numpy as np
from MIOU_ahmed import *
from TGS_Dataset_Preparation import *
from Network_Arcitecture import nn_model
from Weight_Initializer import Initialize_Weights
from lovasz_losses_tf import lovasz_hinge
from early_stopping_on_iou import *


epochs = 200

t = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

tf.reset_default_graph()

im_path = 'C:/Users/MSabry/Desktop/New folder/images'
mask_path = 'C:/Users/MSabry/Desktop/New folder/masks'

batch_size = 15

aug_instance = write_augmented_data(im_path,mask_path)

#aug_data = aug_instance.Augment_data()

train, test = aug_instance.create_dataframe()
print(len(train))
print(len(train)//batch_size)

train_x = train.img_values.values.tolist()
train_y = train.mask_values.values.tolist()

test_x = test.img_values.values.tolist()
test_y = test.mask_values.values.tolist()


X = tf.placeholder( dtype = tf.float32, name = 'X',shape = [batch_size,128,128,1])
Y = tf.placeholder(dtype = tf.float32, name = 'Y',shape =[batch_size,101,101,1])
keep_prob = tf.placeholder(dtype = tf.float32, name = 'keep_prob')
training_flag = tf.placeholder(dtype = tf.bool, name = 'training_flag')
decay = tf.placeholder(dtype = tf.float32, name = 'decay')

weights,biases = Initialize_Weights()

prediction = nn_model(X, weights, biases, batch_size, training_flag = training_flag, 
                      decay = decay, keep_prob = keep_prob)
seg_pred = tf.nn.sigmoid(prediction)
labels = Y

learning_rate = tf.Variable(0.005, name = 'learning_rate', trainable = False)

cost = lovasz_hinge(prediction, labels,per_image=True)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


epoch_loss_list = []
test_loss_list = []

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep=1)
    
    tf.add_to_collection('cost', cost)
    
    early_stopper = Early_Stopping(sess, saver, 30)
    
    for epoch in range(epochs):
        
        epoch_iou_list = []
        epoch_kiou_list = []
        test_iou_list = []
        test_kiou_list = []
        
        epoch_loss = 0
        
        for i in range(len(train)//batch_size):
            
            start = i * batch_size
            end = start + batch_size
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            
            _, loss, s_pred, y_labels = sess.run([optimizer, cost, seg_pred, labels]
            ,feed_dict = {'X:0':batch_x,'Y:0':batch_y, 'training_flag:0': True, 'decay:0':0.9, 'keep_prob:0':1})
            
            iou = MIOU(s_pred, y_labels, 0.4, batch_size)
            epoch_iou_list.extend(iou)
            
            kiou = Kaggle_MIOU(s_pred, y_labels, t, batch_size)
            epoch_kiou_list.extend(kiou)
                        
            batch_loss = loss
            epoch_loss += batch_loss
            
            print('batch {} finished out of {} batches, epoch {}'.format(i+1,len(train)//batch_size, epoch+1))
            print('batch_loss: ',batch_loss)
       
        epoch_loss_list.append(epoch_loss)
        
        
        test_loss = 0
            
        for i in range(len(test)//batch_size):
            
            start = i * batch_size
            end = start + batch_size
            test_batch_x = np.array(test_x[start:end])
            test_batch_y = np.array(test_y[start:end])
            
            t_loss, s_test_pred, y_test_labels = sess.run([cost, seg_pred, labels]
            ,feed_dict = {'X:0':test_batch_x, 'Y:0':test_batch_y, 'training_flag:0': True, 'decay:0':0.9, 'keep_prob:0':1})
            
            test_iou = MIOU(s_test_pred, y_test_labels, 0.4, batch_size)
            test_iou_list.extend(test_iou)
            
            test_kiou = Kaggle_MIOU(s_test_pred, y_test_labels, t, batch_size)
            test_kiou_list.extend(test_kiou)
            
            test_loss+= t_loss
            
        test_loss_list.append(test_loss)
        
        
        early_stopper.save_best_model(np.mean(test_iou_list))
        
        if early_stopper.counter > early_stopper.epochs_to_wait:
            break
            
        print('epoch: {} finished of {} epochs'.format(epoch+1, epochs))
        print('epoch_loss: ', epoch_loss)
        print('epoch iou:  ', np.mean(epoch_iou_list))
        print('epoch kiou:  ', np.mean(epoch_kiou_list))
        print('test_loss', test_loss)
        print('test iou:  ', np.mean(test_iou_list))
        print('test kiou:  ', np.mean(test_kiou_list))
    

