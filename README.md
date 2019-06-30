# TGS-Salt-Identification-Challenge
Semantic segmentation project aims to generate mask images 
Input

Input: 128x128,  grey scale image.

Augmentations

Brightness , vflip, hflip, scale, rotate, horizontal_shear. However, using heavy augmentations could cause problems while training

Base model

U-net model. with last block modified by valid convolutions followed by a 1×1 convolution to squeeze channels and end with a 101×101×1 image. 

ScSE (Spatial-Channel Squeeze & Excitation) both in Encoder and Decoder

dropouts. That was cruel because speeded up the training and improved final result. We even tried to use only 16 filters but ended up using 32.

Learning

Batch size =20, Adam, Cyclic learning rate (mode='triangular2', baselr=1e-4, maxlr=3e-4, step_size=1500), heavy snapshot assembling (averaging last 10 best models with exponentially decreasing weights). Using snapshot assembling made useless blending models or for such lb score lower models didn't give any boost.
