#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103

'''
Just want to load image via python3.5
'''

# import image module
from PIL import Image

# store Image
side = 28
img = [[0]*side for i in range(side)]

# url
gUrl = "C:\\Users\\Yufei\\Desktop\\deep learning\\"
lUrl = "program assignment 2\\data_prog2\\"
sTest = "test_data\\"
sTrain = "train_data\\"

# image number
nImage = 0
suffix = ".jpg"
for im in range(10):
    # construct string for image
    nImage = im+1
    sImage = str(nImage).zfill(5)
    # open image file
    img = Image.open(gUrl+lUrl+sTrain+sImage+suffix)

# handle the image
img.show()

