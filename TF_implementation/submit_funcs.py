import cv2
import numpy as np
import pandas as pd
import os
#import random

mask_path = r'E:\Mohammed\IT\machine learning course\ml projects\TGS Salt Identification Challenge\Data\Train\masks'
images_path = r'E:\Mohammed\IT\machine learning course\ml projects\TGS Salt Identification Challenge\Data\Train\images'
depths_path = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/depths.csv'
depths = pd.read_csv(depths_path)

depths_dict = {}

for i in depths.values:
    depths_dict[i[0]] = i[1]
    
def training_data_generator(mask_path, train_path, img_size= 192, padding = 16):
    
    Dataset = []
    print('Data preparation has started')
    for img in os.listdir(train_path):
        
        original_img_101 = cv2.imread(os.path.join(train_path, img), 0)
        original_mask = cv2.imread(os.path.join(mask_path, img), 0)
        original_mask[original_mask>0]= 1
        original_img_192 = cv2.resize(original_img_101.copy(), (img_size, img_size))
        original_img_224 = cv2.copyMakeBorder(original_img_192.copy(), padding, padding, padding, padding, 0, value= 0)
        img = img.replace('.png', '')
        Dataset.append([img, original_img_224, original_mask])

    print('Data preparation has ended')
    return Dataset



def test_data_generator(path, img_size= 192, padding = 16):
    
    Dataset = []
    print('Data preparation has started')
    for img in os.listdir(path):
        
        original_img_101 = cv2.imread(os.path.join(path, img), 0)
        original_img_192 = cv2.resize(original_img_101.copy(), (img_size, img_size))
        original_img_224 = cv2.copyMakeBorder(original_img_192.copy(), padding, padding, padding, padding, 0, value= 0)
        Dataset.append([img[:-4], original_img_224])

    print('Data preparation has ended')
    return Dataset



def encode_rle(label, batch = True ,batch_size = 1):
    if batch:
        result = []
        for ii in range(batch_size):
            lab = label.copy()[ii]
            _label = rle_encoding(lab)
            result.append(_label)
        return result
    else:
        label_ = rle_encoding(label)
        return label_


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return str(run_lengths).replace('[', '').replace(']', '')

