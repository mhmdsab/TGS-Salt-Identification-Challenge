# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:53:14 2019

@author: User
"""

import pandas as pd
import numpy as np
import cv2
from skimage.util import pad, crop
from skimage.transform import resize
from skimage.io import imshow
from utils import classify_depth, cov_to_class, vertical_mask, small_mask, large_mask, empty_mask
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input


class prepare_data:
    
    def __init__(self,imgs_folder, mask_folder, test_folder, train_df, depth_df):
        
        self.imgs_folder = imgs_folder
        self.mask_folder = mask_folder
        self.test_folder = test_folder
        self.train_df = train_df
        self.depth_df = depth_df
        self.train_set, self.val_set, self.test_set = self.prepare_dataframe()
        
        
    def prepare_dataframe(self):
        
        train = pd.read_csv(self.train_df, index_col = 0)
        depth = pd.read_csv(self.depth_df, index_col = 0)
        
        train_1 = pd.merge(train, depth, on = 'id')
        train_1['depth_class'] = train_1['z'].map(classify_depth)
        train_1['mask_class'] = train_1.index.map(lambda x:cov_to_class(img_id=x,mask_folder = self.mask_folder))
        train_1['vertical_mask'] = train_1.index.map(lambda x:vertical_mask(mask_id=x,mask_folder = self.mask_folder))
        train_1['small_mask'] = train_1.index.map(lambda x:small_mask(mask_id=x,mask_folder = self.mask_folder))
        train_1['large_mask'] = train_1.index.map(lambda x:large_mask(mask_id=x,mask_folder = self.mask_folder))
        train_1['empty_mask'] = train_1.index.map(lambda x:empty_mask(mask_id=x,mask_folder = self.mask_folder))
        
        #removing vertical, small & large masks
        train_1 = train_1[train_1.vertical_mask == 0]
        train_1 = train_1[train_1.large_mask == 0]
        train_1 = train_1[train_1.small_mask == 0]
        
        train_set, val_set = train_test_split(train_1, test_size=0.2, random_state=0,stratify = train_1[['mask_class','depth_class']])
        
        test_set = depth[~depth.index.isin(train.index)]
        return train_set, val_set, test_set


    def train_data_gen(self):
        
        train_imgs = [cv2.imread(self.imgs_folder+'/'+im_id+'.png',1) for im_id in self.train_set.index.values]
        train_imgs = np.array([preprocess_input(pad(i, ((13,14), (13,14), (0,0)), 'reflect')) for i in train_imgs]).astype(np.float32)
        train_masks = [cv2.imread(self.mask_folder+'/'+im_id+'.png',0) for im_id in self.train_set.index.values]
        train_masks = [pad(i, (13,14), 'reflect') for i in train_masks]
        mask128 = np.array(train_masks).astype(np.float32)/255
        mask128 = np.reshape(mask128,(-1,128,128,1))
        mask64 = np.array([resize(i, (64,64),order = 3, preserve_range=True) for i in train_masks]).astype(np.float32)/255
        mask64 = np.reshape(mask64,(-1,64,64,1))
        mask32 = np.array([resize(i, (32,32),order = 3, preserve_range=True) for i in train_masks]).astype(np.float32)/255
        mask32 = np.reshape(mask32,(-1,32,32,1))
        mask16 = np.array([resize(i, (16,16),order = 3, preserve_range=True) for i in train_masks]).astype(np.float32)/255
        mask16 = np.reshape(mask16,(-1,16,16,1))
        empty_mask = self.train_empty_mask_gen()
        
        return train_imgs, [mask128, mask128, mask64, mask32, mask16, empty_mask]
    

    def val_data_gen(self):
        
        val_imgs = [cv2.imread(self.imgs_folder+'/'+im_id+'.png',1) for im_id in self.val_set.index.values]
        val_imgs = np.array([preprocess_input(pad(i, ((13,14), (13,14), (0,0)), 'reflect')) for i in val_imgs]).astype(np.float32)
        val_masks = [cv2.imread(self.mask_folder+'/'+im_id+'.png',0) for im_id in self.val_set.index.values]
        val_masks = [pad(i, (13,14), 'reflect') for i in val_masks]
        mask128 = np.array(val_masks).astype(np.float32)/255
        mask128 = np.reshape(mask128,(-1,128,128,1))
        mask64 = np.array([resize(i, (64,64),order = 3, preserve_range=True) for i in val_masks]).astype(np.float32)/255
        mask64 = np.reshape(mask64,(-1,64,64,1))
        mask32 = np.array([resize(i, (32,32),order = 3, preserve_range=True) for i in val_masks]).astype(np.float32)/255
        mask32 = np.reshape(mask32,(-1,32,32,1))
        mask16 = np.array([resize(i, (16,16),order = 3, preserve_range=True) for i in val_masks]).astype(np.float32)/255
        mask16 = np.reshape(mask16,(-1,16,16,1))
        empty_mask = self.val_empty_mask_gen()
        
        return val_imgs, [mask128, mask128, mask64, mask32, mask16, empty_mask]
    
    
    def test_data_gen(self):
        
        test_imgs = [cv2.imread(self.test_folder+'/'+im_id+'.png',1) for im_id in self.test_set.index.values]
        test_imgs = np.array([preprocess_input(pad(i, ((13,14), (13,14), (0,0)), 'reflect')) for i in test_imgs]).astype(np.float32)
        return test_imgs
    
    
    def train_empty_mask_gen(self):
        return self.train_set.empty_mask.values
    
    
    def val_empty_mask_gen(self):
        return self.val_set.empty_mask.values











