# -*- coding: utf-8 -*-
"""
@author: MSabry
"""
import numpy as np
import pandas as pd

from keras.models import load_model

from metrics import my_iou_metric
from data_preparation import prepare_data
from utils import rle_encoding
from skimage.util import crop

imgs_folder = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/Train/images'
mask_folder = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/Train/masks'
test_folder = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/Test/images'

train = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/train.csv'
depth = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/depths.csv'


inst = prepare_data(imgs_folder, mask_folder, test_folder, train, depth)

test_data = inst.test_data_gen()

final_model = load_model('U-resnet_decoding', custom_objects={'my_iou_metric': my_iou_metric})

def predict_results(model, test_data, inst):
    preds = model.predict(test_data)
    final_preds = preds[1]
    final_pred = np.array([crop(final_preds[i], ((13,14), (13,14), (0,0))).reshape((101,101)) for i in range(18000)])
    final_p = np.where(final_pred>=0.5, 1, 0)
    submit = [rle_encoding(final_p[i]) for i in range(18000)]
    submit_df = pd.DataFrame()
    submit_df['id'] = inst.test_set.index.values
    submit_df['rle_mask'] = submit
    submit_df.to_csv('submission.csv', index = False)
    
    return submit

results = predict_results(final_model, test_data, inst)

