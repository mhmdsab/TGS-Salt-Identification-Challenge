# TGS-Salt-Identification-Challenge
Semantic segmentation project aims to generate mask images identifying salt deposits through a deep learning model.  
---------------------------------------TF_implementation--------------------------------------

Input: 128x128,  grey scale image.

Augmentations

Brightness , vflip, hflip

Base model

U-net model, with last block modified by several valid convolutions followed by a 1×1 convolution to squeeze tensors' channels and end with a 101×101×1 image. 

ScSE (Spatial-Channel Squeeze & Excitation) both in Encoder and Decoder

dropout was applied in the second stage of training with keep prob value 0.5.

Learning

Batch size =15 (maximum available batch size), Adam optimizer, evaluation metric is mean intersection over union, learning rate assigned manually at every training stage.

---------------------------------------Keras implementation--------------------------------------

Input: 101x101,  RGB scale image padded with reflect mode to 128x128.


Base model

U-net model, resnet34 encoder with last block excited with feature pyramid attention network 


ScSE (Spatial-Channel Squeeze & Excitation) in Decoder

dropout was applied in the second stage of training with keep prob value 0.5.

Learning

Batch size =12 (maximum available batch size), Adam optimizer, evaluation metric is mean intersection over union, learning rate reduced on plateau.

better results than tensorflow (0.84 vs 0.76)
