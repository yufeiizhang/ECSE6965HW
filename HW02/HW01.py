#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103

'''
Program Assigment 2
1st question
'''

# import image module
from PIL import Image
import tensorflow as tf

# prepare urls
gUrl = "C:\\Users\\Yufei\\Desktop\\deep learning\\"
lUrl = "program assignment 2\\data_prog2\\"
sTest = "test_data\\"
sTrain = "train_data\\"
suffix = ".jpg"

# load data from training dataset
nImgTrain = 25112
# nImgTrain = 4
imgTrain = []
for im in range(nImgTrain):
    # construct string for image
    nImage = im+1
    sImage = str(nImage).zfill(5)
    # open image file
    imgIm = Image.open(gUrl+lUrl+sTrain+sImage+suffix)
    imgIm = imgIm.convert("L")
    # convert to list
    img1D = imgIm.getdata()
    # append to train dataset
    imgTrain.append(img1D)

with sess in tf.Session:
    


