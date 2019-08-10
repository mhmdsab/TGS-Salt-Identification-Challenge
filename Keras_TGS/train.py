# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:15:30 2019

@author: User
"""
import numpy as np

import tensorflow as tf
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from data_preparation import prepare_data
from layer_extraction import extract_outputs
from MIOU import MIOU
from losses import binary_crossentropy
from segmentation_models import Unet
from metrics import my_iou_metric


imgs_folder = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/Train/images'
mask_folder = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/Train/masks'
test_folder = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/Test/images'

train = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/train.csv'
depth = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/depths.csv'

inst = prepare_data(imgs_folder, mask_folder, test_folder, train, depth)

train_x, train_y = inst.train_data_gen()
val_x, val_y = inst.val_data_gen()

model = Unet('resnet34',input_shape = (128,128,3), encoder_weights='imagenet',
             decoder_filters=(256, 128, 64, 32),FPA = True, SCSE = True, activation = 'sigmoid')

extracted_outputs = extract_outputs(model,FPA = True, SCSE = True)
final_model = Model(model.input, extracted_outputs)

opt = Adam(lr = 0.01)

weights = [ 0.1, 1, 0.1, 0.1, 0.1, 0.1]
sum_weights = np.sum(weights)
loss_weights = [w/sum_weights for w in weights]

metric = {'sigmoid':my_iou_metric,'activated_128':my_iou_metric,'output64':my_iou_metric,
          'output32':my_iou_metric,'output16':my_iou_metric,'output_class':'accuracy'}

cost = [binary_crossentropy, binary_crossentropy, binary_crossentropy,
        binary_crossentropy, binary_crossentropy, binary_crossentropy]

final_model.compile(loss=cost, optimizer= opt, metrics=metric, loss_weights=loss_weights)
#
early_stopping = EarlyStopping(monitor= 'val_activated_128_my_iou_metric', mode = 'max',patience=20, verbose=1)

model_checkpoint = ModelCheckpoint('U-resnet_decoding',monitor = 'val_activated_128_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_activated_128_my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

history = final_model.fit(train_x, train_y,
                    validation_data=[val_x, val_y], 
                    epochs=100,
                    batch_size=12,
                    callbacks=[model_checkpoint, model_checkpoint, reduce_lr])

