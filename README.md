# TGS-Salt-Identification-Challenge
Semantic segmentation project aims to generate mask images identifying salt deposits through a deep learning model.  
Input

Input: 128x128,  grey scale image.

Augmentations

Brightness , vflip, hflip, scale, rotate, horizontal_shear. However, using heavy augmentations could cause problems while training

Base model

U-net model, with last block modified by several valid convolutions followed by a 1×1 convolution to squeeze tensors' channels and end with a 101×101×1 image. 

ScSE (Spatial-Channel Squeeze & Excitation) both in Encoder and Decoder

dropout was applied in the second stage of training with keep prob value 0.5.

Learning

Batch size =15 (maximum available batch size), Adam optimizer, evaluation metric is mean intersection over union, learning rate assigned manually at every training stage.
