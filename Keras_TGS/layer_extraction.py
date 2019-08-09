# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 03:57:53 2019

@author: User
"""
from keras.layers import GlobalAveragePooling2D, Multiply, Add, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
import keras.backend as K

def extract_outputs(model, SCSE = True, FPA = True):
    if SCSE:
        supervision_layers = ['decoder_stage%d_scSE' % i for i in range(2, -1, -1)]
    else: 
        supervision_layers = ['decoder_stage%d_relu2' % i for i in range(2, -1, -1)]
        
    if FPA:
        middle_layer_name = 'bottle_neck'
    else: 
        middle_layer_name = 'relu1'
    
    #importing raw output layer
    raw_output = model.output
    
    #importing down masks for deep supervision 
    output_names = ['output64', 'output32', 'output16']
    output_64, output_32, output_16 = [Conv2D(1,(1,1), padding="same", activation='sigmoid', name = n) (model.get_layer(l).output) for n,l in zip(output_names, supervision_layers)]
    
    #importing middle layer
    middle_layer = model.get_layer(middle_layer_name).output
    
    #empty class classifier
    empty_class = GlobalAveragePooling2D()(middle_layer)
    empty_class = Dense(1, activation='sigmoid', name = 'output_class')(empty_class)
    
    #getting final_output
    final_output = Multiply(name = 'activated_128')([raw_output, empty_class])
    
    return [raw_output, final_output, output_64, output_32, output_16, empty_class]
    
