from TGS_data_augmentation import *
from sklearn.model_selection import train_test_split
import scipy.misc
import numpy as np
import pandas as pd
import os
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

        
class write_augmented_data:
    
    def __init__(self,img_path,mask_path):
        self.img_path = img_path
        self.mask_path = mask_path
        self.imgs_list = os.listdir(img_path)[:-1]
        self.masks_list = os.listdir(mask_path)[:-1]
    
    def load_data(self):
       
       imgs = [cv2.imread(os.path.join(self.img_path,self.imgs_list[i]),1)/255
       for i in range(len(self.masks_list))]
       
       masks = [cv2.imread(os.path.join(self.mask_path,self.masks_list[i]),0)/255
       for i in range(len(self.masks_list))]
       
       return imgs,masks
   
    @staticmethod
    def cov_to_class(val):
        for i in range(0, 10):
            if val * 0.00098 <= i:
                return i
            
    @staticmethod
    def classify_depth(depth):
        class_values = np.linspace(50,959,6)
        for i in range(len(class_values)):
            if depth <= class_values[i]:
                return i

            
    @staticmethod
    def resize(img):
        return cv2.resize(img,(224,224))
    
 
    @staticmethod
    def add_mask_channel(mask):
        h,w = mask.shape
        return np.reshape(mask,(h,w,1))
   
    
    def Augment_data(self):
        
        img,mask = self.load_data()
        
        for i in range(len(mask)):
            rand = np.random.choice(3)
            if rand == 0:
                #horizontally flipped images
                hflip_img,hflipd_mask = hflip(img[i],mask[i])
                scipy.misc.imsave(os.path.join(self.img_path,'hflipped_'+self.imgs_list[i]),hflip_img)
                scipy.misc.imsave(os.path.join(self.mask_path,'hflipped_'+self.masks_list[i]),hflipd_mask)


            elif rand == 1:
                #vertically flipped images
                vflip_img,vflipd_mask = vflip(img[i],mask[i])
                scipy.misc.imsave(os.path.join(self.img_path,'vflipped_'+self.imgs_list[i]),vflip_img)
                scipy.misc.imsave(os.path.join(self.mask_path,'vflipped_'+self.masks_list[i]),vflipd_mask)

                
            elif rand == 2:
                #horizontally and vertically flipped images
                hvflip_img,hvflipd_mask = hvflip(img[i],mask[i])
                scipy.misc.imsave(os.path.join(self.img_path,'hvflipped_'+self.imgs_list[i]),hvflip_img)
                scipy.misc.imsave(os.path.join(self.mask_path,'hvflipped_'+self.masks_list[i]),hvflipd_mask)

            
#            elif rand == 3:
#                #transposed images
#                trans_img, trans_mask = transpose(img[i], mask[i])
#                print(trans_mask.shape)
#                print(trans_img.shape)
#                scipy.misc.imsave(os.path.join(self.img_path,'transposed_'+self.imgs_list[i]), trans_img)
#                scipy.misc.imsave(os.path.join(self.mask_path,'transposed_'+self.masks_list[i]), trans_mask)

                
            #rescaled crops
            sizes = [512,480,384,256]
            cropped_img, cropped_mask = rescaled_crops(img[i], mask[i],np.random.choice(sizes))
            cropped_img = cv2.resize(cropped_img,(101,101))
            cropped_mask = cv2.resize(cropped_mask,(101,101))
            scipy.misc.imsave(os.path.join(self.img_path,'cropped_'+self.imgs_list[i]),cropped_img)
            scipy.misc.imsave(os.path.join(self.mask_path,'cropped_'+self.masks_list[i]),cropped_mask)
            
            #padded images
            padded_img, padded_mask = do_center_pad_to_factor2(img[i], mask[i])
            rand1 = np.random.choice(2)
            
            if rand1 == 0:
                padded_img = do_invert_intensity(padded_img)
                padded_img = cv2.resize(padded_img,(101,101))
                padded_mask = cv2.resize(padded_mask,(101,101))
                scipy.misc.imsave(os.path.join(self.img_path,'padded-inv-intensity_'+self.imgs_list[i]),padded_img)
                scipy.misc.imsave(os.path.join(self.mask_path,'padded-inv-intensity_'+self.masks_list[i]),padded_mask)
                
            if rand1 == 1:
                padded_img = do_brightness_shift(padded_img,alpha=np.random.choice([-0.3,-0.1,0.1,0.3]))
                scipy.misc.imsave(os.path.join(self.img_path,'padded-brightness-shift_'+self.imgs_list[i]),padded_img)
                scipy.misc.imsave(os.path.join(self.mask_path,'padded-brightness-shift_'+self.masks_list[i]),padded_mask)
            
                
            rand2 = np.random.choice(2)

            if rand2 == 0:
                transformed_img, transformed_mask = do_shift_scale_rotate2(img[i], mask[i])
                scipy.misc.imsave(os.path.join(self.img_path,'shift-scale-rotated_'+self.imgs_list[i]),transformed_img)
                scipy.misc.imsave(os.path.join(self.mask_path,'shift-scale-rotated_'+self.masks_list[i]),transformed_mask)
                
            elif rand2 == 1:
                sheared_img, sheared_mask = do_horizontal_shear2(img[i], mask[i],dx=0.2)
                scipy.misc.imsave(os.path.join(self.img_path,'sheared_'+self.imgs_list[i]),sheared_img)
                scipy.misc.imsave(os.path.join(self.mask_path,'sheared_'+self.masks_list[i]),sheared_mask)
    
 
    def create_dataframe(self):
        
        imgs_list = os.listdir(self.img_path)
        if 'Thumbs.db' in imgs_list:
            imgs_list.remove('Thumbs.db')
            
        masks_list = os.listdir(self.mask_path)
        if 'Thumbs.db' in masks_list:
            masks_list.remove('Thumbs.db')
        
        df = pd.DataFrame()

        df['imgs'] = imgs_list
        df['masks'] = masks_list
       
        df['img_values'] = [cv2.imread(os.path.join(self.img_path,i),1)/255 for i in imgs_list]
        df['mask_values'] = [cv2.imread(os.path.join(self.mask_path,i),0)/255 for i in masks_list]
        
        df['coverage'] = df.mask_values.map(np.sum)
        df['mask_class'] = df.coverage.map(write_augmented_data.cov_to_class)
        
        df['img_values'] = df.img_values.map(write_augmented_data.resize)
        df['mask_values'] = df.mask_values.map(write_augmented_data.add_mask_channel)
        
        df['imgs'] = df['imgs'].str.split('.').str.get(0).str.split('_').str.get(-1)
        df['masks'] = df['masks'].str.split('.').str.get(0).str.split('_').str.get(-1)
        
        depth_df = pd.read_csv('C:/Users/MSabry/Desktop/New folder/depths.csv')
        
        df_z = pd.merge(df,depth_df,left_on = 'imgs',right_on = 'id',how = 'left')

        df_z['depth_class'] = df_z.z.map(write_augmented_data.classify_depth)
        
#        df_z.to_csv('C:/Users/User/Desktop/testing/df_z.csv')
        
        train_df, test_df = train_test_split(df_z, test_size=0.2, random_state=0,stratify = df_z[['mask_class','depth_class']])
        
        return train_df, test_df
    
        
        


#im_path = 'C:/Users/User/Desktop/testing/images'
#mask_path = 'C:/Users/User/Desktop/testing/masks'
#
#a = write_augmented_data(im_path,mask_path)
#b = a.Augment_data()
#
#train,test = a.create_dataframe()

