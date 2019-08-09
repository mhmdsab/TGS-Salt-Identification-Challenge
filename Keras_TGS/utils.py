# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:00:28 2019

@author: User
"""
import numpy as np
import cv2


def cov_to_class(img_id, mask_folder):
    img = cv2.imread(mask_folder+'/'+img_id+'.png',0).astype(np.float32)/255
    val = np.sum(img) / (img.shape[0] * img.shape[1])
    for i in range(0, 11):
        if val * 10 <= i :
            return i

        
def classify_depth(depth):
    class_values = np.linspace(50,959,6)
    for i in range(len(class_values)):
        if depth <= class_values[i]:
            return i
        

def vertical_mask(mask_id, mask_folder):
    mask = cv2.imread(mask_folder+'/'+mask_id+'.png',0).astype(np.float32)/255
    res = np.sum(mask, axis=0)
    res = np.unique(res)
    return 1 * (len(res) == 2 and res[0] == 0 and res[1] == mask.shape[0])



def small_mask(mask_id, mask_folder, thr=0.005):
    mask = cv2.imread(mask_folder+'/'+mask_id+'.png',0).astype(np.float32)/255
    res = np.mean(mask)
    return 1 * ((res < thr) and (res > 0))



def large_mask(mask_id, mask_folder, thr=0.997):
    mask = cv2.imread(mask_folder+'/'+mask_id+'.png',0).astype(np.float32)/255
    res = np.mean(mask)
    return 1 * ((res > thr) and (res < 1))



def empty_mask(mask_id, mask_folder):
    mask = cv2.imread(mask_folder+'/'+mask_id+'.png',0).astype(np.float32)/255
    res = np.sum(mask)
    if res == 0:
        return 1
    else:
        return 0


def add_channel(img):
    h,w = img.shape
    return np.reshape(img,(h,w,1))


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