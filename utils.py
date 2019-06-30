import tensorflow as tf


def convolve(layer, kernel, padding = 'SAME', strides = [1,1,1,1], name = '_'):
    return tf.nn.conv2d(layer, filter = kernel, strides = strides, padding = padding, name = name)


def convolve_T(layer, kernel, output_shape, strides = [1,2,2,1]):
    return tf.nn.conv2d_transpose(layer, filter = kernel, strides = strides, output_shape = output_shape)


def MaxPool2d(layer, ksize = [1,2,2,1] ,strides = [1,2,2,1], padding = 'VALID'):
    return tf.nn.max_pool(layer, ksize = ksize, strides = strides, padding = padding)


def convolution_block(input_layer, conv_kernel, bias, training_flag, decay, activate = True, padding = 'SAME'):
    x = convolve(input_layer, conv_kernel, padding = padding)
    x = tf.nn.bias_add(x, bias)
    
    if activate:
        x = tf.contrib.layers.batch_norm(x, is_training = training_flag, decay = decay, zero_debias_moving_mean=True)
        x = tf.nn.relu(x)
    return x


def residual_block(input_layer, conv_kernel1, conv_kernel2, conv_kernel3, bias1, bias2, bias3, 
                   training_flag, decay, keep_prob , batch_activate = False, padding = 'SAME', drop = False, 
                   do_res = True ):
    
    x1 = convolution_block(input_layer, conv_kernel1, bias1, training_flag, decay, padding = padding, activate = False)
    if drop:
        x1 = tf.nn.dropout(x1, keep_prob = keep_prob)
    x = convolution_block(x1, conv_kernel2, bias2, training_flag, decay, padding = padding)
    x = convolution_block(x, conv_kernel3, bias3, training_flag, decay, padding = padding)
    if do_res:
        x = tf.add(x1, x)
    if batch_activate:
        x = tf.contrib.layers.batch_norm(x, is_training = training_flag, decay = decay, zero_debias_moving_mean=True)
    return x



def crop(tensor_to_be_cropped, tensor_to_be_concatenated):
    l1 = tensor_to_be_cropped.get_shape()[1]
    l2 = tensor_to_be_concatenated.get_shape()[1]
    
    if l2 % 2 == 0 :
        margin = l2//2
        start = l1//2 - margin
        end = l1//2 + margin
   
    else:
        margin = l2//2
        start = l1//2 - margin 
        end = l1//2 + margin +1

    print(start)
    print(end)
    
    cropped_tensor = tensor_to_be_cropped[:,start:end,start:end,:]
    return cropped_tensor

def concatenate(tensor_to_be_cropped, tensor_to_be_concatenated):
    cropped_tensor = crop(tensor_to_be_cropped, tensor_to_be_concatenated)
    concatenated_tensor = tf.concat([tensor_to_be_concatenated, cropped_tensor], axis = 3)
    
    return concatenated_tensor



    