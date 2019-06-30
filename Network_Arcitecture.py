import tensorflow as tf
from utils import *
from s_c_S_E import *


def nn_model(data, weights, biases, batch_size, training_flag, decay, keep_prob):
    
    with tf.variable_scope('block_1'):
        
        res_block_1_a = residual_block(data, weights['res_block_1_a_1_filter'], weights['res_block_1_a_2_filter'], 
                                       weights['res_block_1_a_3_filter'], biases['res_block_1_a_1_bias'], 
                                       biases['res_block_1_a_2_bias'], biases['res_block_1_a_3_bias'], training_flag, decay, keep_prob = keep_prob)
        
        res_block_1_a = scSE(res_block_1_a)._scSE_()
   
        res_block_1_b = residual_block(res_block_1_a, weights['res_block_1_b_1_filter'], weights['res_block_1_b_2_filter'], 
                                       weights['res_block_1_b_3_filter'], biases['res_block_1_b_1_bias'], 
                                       biases['res_block_1_b_2_bias'], biases['res_block_1_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_1_b = scSE(res_block_1_b)._scSE_()
        
        res_block_1_b = tf.nn.dropout(res_block_1_b, keep_prob = keep_prob)
       

    with tf.variable_scope('block_2'):
        
        layer_2 = MaxPool2d(res_block_1_b)
        
        layer_2 = scSE(layer_2)._scSE_()
        

        res_block_2_a = residual_block(layer_2, weights['res_block_2_a_1_filter'], weights['res_block_2_a_2_filter'], 
                                       weights['res_block_2_a_3_filter'], biases['res_block_2_a_1_bias'], 
                                       biases['res_block_2_a_2_bias'], biases['res_block_2_a_3_bias'], training_flag, decay, keep_prob = keep_prob)
        
        res_block_2_a = scSE(res_block_2_a)._scSE_()

        res_block_2_b = residual_block(res_block_2_a, weights['res_block_2_b_1_filter'], weights['res_block_2_b_2_filter'], 
                                       weights['res_block_2_b_3_filter'], biases['res_block_2_b_1_bias'], 
                                       biases['res_block_2_b_2_bias'], biases['res_block_2_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_2_b = scSE(res_block_2_b)._scSE_()
        
        res_block_2_b = tf.nn.dropout(res_block_2_b, keep_prob = keep_prob)

      
        
    with tf.variable_scope('block_3'):
        
        layer_3 = MaxPool2d(res_block_2_b)
        
        layer_3 = scSE(layer_3)._scSE_()
        

        res_block_3_a = residual_block(layer_3, weights['res_block_3_a_1_filter'], weights['res_block_3_a_2_filter'], 
                                       weights['res_block_3_a_3_filter'], biases['res_block_3_a_1_bias'], 
                                       biases['res_block_3_a_2_bias'], biases['res_block_3_a_3_bias'], training_flag, decay, keep_prob = keep_prob)
        
        res_block_3_a = scSE(res_block_3_a)._scSE_()

        res_block_3_b = residual_block(res_block_3_a, weights['res_block_3_b_1_filter'], weights['res_block_3_b_2_filter'], 
                                       weights['res_block_3_b_3_filter'], biases['res_block_3_b_1_bias'], 
                                       biases['res_block_3_b_2_bias'], biases['res_block_3_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_3_b = scSE(res_block_3_b)._scSE_()
                
        res_block_3_b = tf.nn.dropout(res_block_3_b, keep_prob = keep_prob)

        
   
    with tf.variable_scope('block_4'):
        
        layer_4 = MaxPool2d(res_block_3_b)
        
        layer_4 = scSE(layer_4)._scSE_()
        

        res_block_4_a = residual_block(layer_4, weights['res_block_4_a_1_filter'], weights['res_block_4_a_2_filter'], 
                                       weights['res_block_4_a_3_filter'], biases['res_block_4_a_1_bias'], 
                                       biases['res_block_4_a_2_bias'], biases['res_block_4_a_3_bias'], training_flag, decay, keep_prob = keep_prob)
        
        res_block_4_a = scSE(res_block_4_a)._scSE_()

        res_block_4_b = residual_block(res_block_4_a, weights['res_block_4_b_1_filter'], weights['res_block_4_b_2_filter'], 
                                       weights['res_block_4_b_3_filter'], biases['res_block_4_b_1_bias'], 
                                       biases['res_block_4_b_2_bias'], biases['res_block_4_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_4_b = scSE(res_block_4_b)._scSE_()
        
        
        res_block_4_b = tf.nn.dropout(res_block_4_b, keep_prob = keep_prob)

   
    with tf.variable_scope('block_5'):
        
        layer_5 = MaxPool2d(res_block_4_b)
        
        layer_5 = scSE(layer_5)._scSE_()

        res_block_5_a = residual_block(layer_5, weights['res_block_5_a_1_filter'], weights['res_block_5_a_2_filter'], 
                                       weights['res_block_5_a_3_filter'], biases['res_block_5_a_1_bias'], 
                                       biases['res_block_5_a_2_bias'], biases['res_block_5_a_3_bias'], training_flag, decay, keep_prob = keep_prob)
        
        res_block_5_a = scSE(res_block_5_a)._scSE_()
        
        res_block_5_b = residual_block(res_block_5_a, weights['res_block_5_b_1_filter'], weights['res_block_5_b_2_filter'], 
                                       weights['res_block_5_b_3_filter'], biases['res_block_5_b_1_bias'], 
                                       biases['res_block_5_b_2_bias'], biases['res_block_5_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_5_b = scSE(res_block_5_b)._scSE_()
    
        
        
    with tf.variable_scope('block_6'):
        
        upsampled_layer_6 = convolve_T(res_block_5_b, weights['block_6_upsample_filter'],
                                         output_shape = [batch_size, 16, 16, first_filter_depth*8])
        
        
        concatenated_layer_6 = tf.concat([upsampled_layer_6,res_block_4_b],axis = 3)
        
        concatenated_layer_6 = scSE(concatenated_layer_6)._scSE_()
        
        concatenated_layer_6 = tf.nn.dropout(concatenated_layer_6, keep_prob = keep_prob)
        
        
        res_block_6_a = residual_block(concatenated_layer_6, weights['res_block_6_a_1_filter'], weights['res_block_6_a_2_filter'], 
                                       weights['res_block_6_a_3_filter'], biases['res_block_6_a_1_bias'], 
                                       biases['res_block_6_a_2_bias'], biases['res_block_6_a_3_bias'], 
                                       training_flag, decay, keep_prob = keep_prob)
        
        res_block_6_a = scSE(res_block_6_a)._scSE_()

        res_block_6_b = residual_block(res_block_6_a, weights['res_block_6_b_1_filter'], weights['res_block_6_b_2_filter'], 
                                       weights['res_block_6_b_3_filter'], biases['res_block_6_b_1_bias'], 
                                       biases['res_block_6_b_2_bias'], biases['res_block_6_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_6_b = scSE(res_block_6_b)._scSE_()
        

        
    with tf.variable_scope('block_7'):
        
        upsampled_layer_7 = convolve_T(res_block_6_b, weights['block_7_upsample_filter'],
                                         output_shape = [batch_size,32,32,first_filter_depth*4])
        
        concatenated_layer_7 = tf.concat([upsampled_layer_7,res_block_3_b],axis = 3)
        
        concatenated_layer_7 = scSE(concatenated_layer_7)._scSE_()
        
        concatenated_layer_7 = tf.nn.dropout(concatenated_layer_7, keep_prob = keep_prob)

        
        
        res_block_7_a = residual_block(concatenated_layer_7, weights['res_block_7_a_1_filter'], weights['res_block_7_a_2_filter'], 
                                       weights['res_block_7_a_3_filter'], biases['res_block_7_a_1_bias'], 
                                       biases['res_block_7_a_2_bias'], biases['res_block_7_a_3_bias'], 
                                       training_flag, decay, keep_prob = keep_prob)
        
        res_block_7_a = scSE(res_block_7_a)._scSE_()
        

        res_block_7_b = residual_block(res_block_7_a, weights['res_block_7_b_1_filter'], weights['res_block_7_b_2_filter'], 
                                       weights['res_block_7_b_3_filter'], biases['res_block_7_b_1_bias'], 
                                       biases['res_block_7_b_2_bias'], biases['res_block_7_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_7_b = scSE(res_block_7_b)._scSE_()
        


    with tf.variable_scope('block_8'):
        
        upsampled_layer_8 = convolve_T(res_block_7_b, weights['block_8_upsample_filter'],
                                         output_shape = [batch_size,64,64,first_filter_depth*2])
        
        concatenated_layer_8 = tf.concat([upsampled_layer_8,res_block_2_b],axis = 3)
        
        concatenated_layer_8 = scSE(concatenated_layer_8)._scSE_()
        
        concatenated_layer_8 = tf.nn.dropout(concatenated_layer_8, keep_prob = keep_prob)

        
        res_block_8_a = residual_block(concatenated_layer_8, weights['res_block_8_a_1_filter'], weights['res_block_8_a_2_filter'], 
                                       weights['res_block_8_a_3_filter'], biases['res_block_8_a_1_bias'], 
                                       biases['res_block_8_a_2_bias'], biases['res_block_8_a_3_bias'], 
                                       training_flag, decay, keep_prob = keep_prob)
        
        res_block_8_a = scSE(res_block_8_a)._scSE_()

        res_block_8_b = residual_block(res_block_8_a, weights['res_block_8_b_1_filter'], weights['res_block_8_b_2_filter'], 
                                       weights['res_block_8_b_3_filter'], biases['res_block_8_b_1_bias'], 
                                       biases['res_block_8_b_2_bias'], biases['res_block_8_b_3_bias'], training_flag, decay,
                                       batch_activate = True, keep_prob = keep_prob)
        
        res_block_8_b = scSE(res_block_8_b)._scSE_()
        
        
    with tf.variable_scope('block_9'):
        
        upsampled_layer_9 = convolve_T(res_block_8_b, weights['block_9_upsample_filter'],
                                         output_shape = [batch_size,128,128,first_filter_depth])
        
        concatenated_layer_9 = tf.concat([res_block_1_b, upsampled_layer_9], axis = 3)
        
        concatenated_layer_9 = scSE(concatenated_layer_9)._scSE_()
        
        concatenated_layer_9 = tf.nn.dropout(concatenated_layer_9, keep_prob = keep_prob)


        res_block_9_a = residual_block(concatenated_layer_9, weights['res_block_9_a_1_filter'], weights['res_block_9_a_2_filter'], 
                                       weights['res_block_9_a_3_filter'], biases['res_block_9_a_1_bias'], 
                                       biases['res_block_9_a_2_bias'], biases['res_block_9_a_3_bias'], training_flag, decay,
                                       padding = 'VALID', keep_prob = keep_prob, do_res = False )
        
        res_block_9_a = scSE(res_block_9_a)._scSE_()
        

        res_block_9_b = residual_block(res_block_9_a, weights['res_block_9_b_1_filter'], weights['res_block_9_b_2_filter'], 
                                       weights['res_block_9_b_3_filter'], biases['res_block_9_b_1_bias'], 
                                       biases['res_block_9_b_2_bias'], biases['res_block_9_b_3_bias'], training_flag, decay,
                                       batch_activate = True, padding = 'VALID', keep_prob = keep_prob, do_res = False )
        
        res_block_9_b = scSE(res_block_9_b)._scSE_()
        
        conv_9_out = convolve(res_block_9_b, weights['block_9_output_filter'],padding = 'VALID',  name = 'conv_9_out')
    
    return conv_9_out