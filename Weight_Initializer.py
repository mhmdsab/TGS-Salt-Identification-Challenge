import tensorflow as tf


input_image_size, input_image_depth = 101, 1

first_filter_depth = 16
f_size = 3
upsamle_filter_size = 2

conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()


def Initialize_Weights():
    weights = {'res_block_1_a_1_filter':tf.Variable(conv_initializer(shape = [7, 7, input_image_depth, first_filter_depth]),
                                      name = 'res_block_1_a_1_filter'),
               
               'res_block_1_a_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth, first_filter_depth]),
                                      name = 'res_block_1_a_2_filter'),

               'res_block_1_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth, first_filter_depth]),
                                      name = 'res_block_1_a_3_filter'),
                                                    
               'res_block_1_b_1_filter':tf.Variable(conv_initializer(shape = [7, 7, first_filter_depth, first_filter_depth]),
                                      name = 'res_block_1_b_1_filter'),
               
               'res_block_1_b_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth, first_filter_depth]),
                                      name = 'res_block_1_b_2_filter'),

               'res_block_1_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth, first_filter_depth]),
                                      name = 'res_block_1_b_3_filter'),
    
               'res_block_2_a_1_filter':tf.Variable(conv_initializer(shape = [7, 7, first_filter_depth, first_filter_depth*2]),
                                      name = 'res_block_2_a_1_filter'),
               
               'res_block_2_a_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_2_a_2_filter'),

               'res_block_2_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_2_a_3_filter'),
                                      
               'res_block_2_b_1_filter':tf.Variable(conv_initializer(shape = [7, 7, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_2_b_1_filter'),
               
               'res_block_2_b_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_2_b_2_filter'),

               'res_block_2_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_2_b_3_filter'),
                                                    
               'res_block_3_a_1_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*4]),
                                      name = 'res_block_3_a_1_filter'),
               
               'res_block_3_a_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*4, first_filter_depth*4]),
                                      name = 'res_block_3_a_2_filter'),

               'res_block_3_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4, first_filter_depth*4]),
                                      name = 'res_block_3_a_3_filter'),
                                      
               'res_block_3_b_1_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*4, first_filter_depth*4]),
                                      name = 'res_block_3_b_1_filter'),
               
               'res_block_3_b_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*4, first_filter_depth*4]),
                                      name = 'res_block_3_b_2_filter'),

               'res_block_3_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4, first_filter_depth*4]),
                                      name = 'res_block_3_b_3_filter'),
                                       
               'res_block_4_a_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4, first_filter_depth*8]),
                                      name = 'res_block_4_a_1_filter'),
               
               'res_block_4_a_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8, first_filter_depth*8]),
                                      name = 'res_block_4_a_2_filter'),

               'res_block_4_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8, first_filter_depth*8]),
                                      name = 'res_block_4_a_3_filter'),
                                      
               'res_block_4_b_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8, first_filter_depth*8]),
                                      name = 'res_block_4_b_1_filter'),
               
               'res_block_4_b_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8, first_filter_depth*8]),
                                      name = 'res_block_4_b_2_filter'),

               'res_block_4_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8, first_filter_depth*8]),
                                      name = 'res_block_4_b_3_filter'),
                                              
               'res_block_5_a_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8, first_filter_depth*16]),
                                      name = 'res_block_5_a_1_filter'),
               
               'res_block_5_a_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*16, first_filter_depth*16]),
                                      name = 'res_block_5_a_2_filter'),

               'res_block_5_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*16, first_filter_depth*16]),
                                      name = 'res_block_5_a_3_filter'),
                                      
               'res_block_5_b_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*16, first_filter_depth*16]),
                                      name = 'res_block_5_b_1_filter'),
               
               'res_block_5_b_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*16, first_filter_depth*16]),
                                      name = 'res_block_5_b_2_filter'),

               'res_block_5_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*16, first_filter_depth*16]),
                                      name = 'res_block_5_b_3_filter'),
                                              
               'block_6_upsample_filter':tf.Variable(conv_initializer(shape = [2, 2, first_filter_depth*8, first_filter_depth*16]),
                                      name = 'block_6_upsample_filter'),
                        
               'res_block_6_a_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8*2, first_filter_depth*8*2]),
                                      name = 'res_block_6_a_1_filter'),
               
               'res_block_6_a_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8*2, first_filter_depth*8*2]),
                                      name = 'res_block_6_a_2_filter'),

               'res_block_6_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8*2, first_filter_depth*8*2]),
                                      name = 'res_block_6_a_3_filter'),
                                      
               'res_block_6_b_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8*2, first_filter_depth*8*2]),
                                      name = 'res_block_6_b_1_filter'),
               
               'res_block_6_b_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8*2, first_filter_depth*8*2]),
                                      name = 'res_block_6_b_2_filter'),

               'res_block_6_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*8*2, first_filter_depth*8*2]),
                                      name = 'res_block_6_b_3_filter'),
                                              
               'block_7_upsample_filter':tf.Variable(conv_initializer(shape = [2, 2, first_filter_depth*4, first_filter_depth*8*2]),
                                      name = 'block_7_upsample_filter'),
                        
               'res_block_7_a_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4*2, first_filter_depth*4*2]),
                                      name = 'res_block_7_a_1_filter'),
               
               'res_block_7_a_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4*2, first_filter_depth*4*2]),
                                      name = 'res_block_7_a_2_filter'),

               'res_block_7_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4*2, first_filter_depth*4*2]),
                                      name = 'res_block_7_a_3_filter'),
                                      
               'res_block_7_b_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4*2, first_filter_depth*4*2]),
                                      name = 'res_block_7_b_1_filter'),
               
               'res_block_7_b_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4*2, first_filter_depth*4*2]),
                                      name = 'res_block_7_b_2_filter'),

               'res_block_7_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*4*2, first_filter_depth*4*2]),
                                      name = 'res_block_7_b_3_filter'),
                                              
               'block_8_upsample_filter':tf.Variable(conv_initializer(shape = [2, 2, first_filter_depth*2, first_filter_depth*4*2]),
                                      name = 'block_8_upsample_filter'),
                        
               'res_block_8_a_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2*2, first_filter_depth*2*2]),
                                      name = 'res_block_8_a_1_filter'),
               
               'res_block_8_a_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2*2, first_filter_depth*2*2]),
                                      name = 'res_block_8_a_2_filter'),

               'res_block_8_a_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2*2, first_filter_depth*2*2]),
                                      name = 'res_block_8_a_3_filter'),
                                      
               'res_block_8_b_1_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2*2, first_filter_depth*2*2]),
                                      name = 'res_block_8_b_1_filter'),
               
               'res_block_8_b_2_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2*2, first_filter_depth*2*2]),
                                      name = 'res_block_8_b_2_filter'),

               'res_block_8_b_3_filter':tf.Variable(conv_initializer(shape = [f_size, f_size, first_filter_depth*2*2, first_filter_depth*2*2]),
                                      name = 'res_block_8_b_3_filter'),
                                              
               'block_9_upsample_filter':tf.Variable(conv_initializer(shape = [2, 2, first_filter_depth, first_filter_depth*2*2]),
                                      name = 'block_9_upsample_filter'),
                        
               'res_block_9_a_1_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_9_a_1_filter'),
               
               'res_block_9_a_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_9_a_2_filter'),

               'res_block_9_a_3_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_9_a_3_filter'),
                                      
               'res_block_9_b_1_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_9_b_1_filter'),
               
               'res_block_9_b_2_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_9_b_2_filter'),

               'res_block_9_b_3_filter':tf.Variable(conv_initializer(shape = [5, 5, first_filter_depth*2, first_filter_depth*2]),
                                      name = 'res_block_9_b_3_filter'),
                                              
               'block_9_output_filter':tf.Variable(conv_initializer(shape = [4, 4, first_filter_depth*2, 1]),
                                      name = 'block_9_output_filter')}
    
    
    biases = {'res_block_1_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth]),name = 'res_block_1_a_1_bias'),
              'res_block_1_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth]),name = 'res_block_1_a_2_bias'),
              'res_block_1_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth]),name = 'res_block_1_a_3_bias'),
              'res_block_1_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth]),name = 'res_block_1_b_1_bias'),
              'res_block_1_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth]),name = 'res_block_1_b_2_bias'),
              'res_block_1_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth]),name = 'res_block_1_b_3_bias'),
              
              'res_block_2_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_2_a_1_bias'),
              'res_block_2_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_2_a_2_bias'),
              'res_block_2_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_2_a_3_bias'),
              'res_block_2_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_2_b_1_bias'),
              'res_block_2_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_2_b_2_bias'),
              'res_block_2_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_2_b_3_bias'),
              
              'res_block_3_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4]),name = 'res_block_3_a_1_bias'),
              'res_block_3_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4]),name = 'res_block_3_a_2_bias'),
              'res_block_3_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4]),name = 'res_block_3_a_3_bias'),
              'res_block_3_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4]),name = 'res_block_3_b_1_bias'),
              'res_block_3_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4]),name = 'res_block_3_b_2_bias'),
              'res_block_3_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4]),name = 'res_block_3_b_3_bias'),
              
              'res_block_4_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8]),name = 'res_block_4_a_1_bias'),
              'res_block_4_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8]),name = 'res_block_4_a_2_bias'),
              'res_block_4_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8]),name = 'res_block_4_a_3_bias'),
              'res_block_4_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8]),name = 'res_block_4_b_1_bias'),
              'res_block_4_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8]),name = 'res_block_4_b_2_bias'),
              'res_block_4_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8]),name = 'res_block_4_b_3_bias'),

              'res_block_5_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*16]),name = 'res_block_5_a_1_bias'),
              'res_block_5_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*16]),name = 'res_block_5_a_2_bias'),
              'res_block_5_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*16]),name = 'res_block_5_a_3_bias'),
              'res_block_5_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*16]),name = 'res_block_5_b_1_bias'),
              'res_block_5_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*16]),name = 'res_block_5_b_2_bias'),
              'res_block_5_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*16]),name = 'res_block_5_b_3_bias'),
              
              'res_block_6_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8*2]),name = 'res_block_6_a_1_bias'),
              'res_block_6_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8*2]),name = 'res_block_6_a_2_bias'),
              'res_block_6_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8*2]),name = 'res_block_6_a_3_bias'),
              'res_block_6_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8*2]),name = 'res_block_6_b_1_bias'),
              'res_block_6_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8*2]),name = 'res_block_6_b_2_bias'),
              'res_block_6_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*8*2]),name = 'res_block_6_b_3_bias'),
              
              'res_block_7_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4*2]),name = 'res_block_7_a_1_bias'),
              'res_block_7_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4*2]),name = 'res_block_7_a_2_bias'),
              'res_block_7_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4*2]),name = 'res_block_7_a_3_bias'),
              'res_block_7_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4*2]),name = 'res_block_7_b_1_bias'),
              'res_block_7_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4*2]),name = 'res_block_7_b_2_bias'),
              'res_block_7_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*4*2]),name = 'res_block_7_b_3_bias'),
              
              'res_block_8_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2*2]),name = 'res_block_8_a_1_bias'),
              'res_block_8_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2*2]),name = 'res_block_8_a_2_bias'),
              'res_block_8_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2*2]),name = 'res_block_8_a_3_bias'),
              'res_block_8_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2*2]),name = 'res_block_8_b_1_bias'),
              'res_block_8_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2*2]),name = 'res_block_8_b_2_bias'),
              'res_block_8_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2*2]),name = 'res_block_8_b_3_bias'),
              
              'res_block_9_a_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_9_a_1_bias'),
              'res_block_9_a_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_9_a_2_bias'),
              'res_block_9_a_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_9_a_3_bias'),
              'res_block_9_b_1_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_9_b_1_bias'),
              'res_block_9_b_2_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_9_b_2_bias'),
              'res_block_9_b_3_bias':tf.Variable(tf.zeros(shape = [first_filter_depth*2]),name = 'res_block_9_b_3_bias')}
    
    return weights,biases
