#!/usr/bin/python
# pylint: disable=I0011,C0103

'''
Just want to load image via python3.5
'''

# import image module
from PIL import Image

# url
gUrl = "C:\\Users\\Yufei\\Desktop\\deep learning\\"
lUrl = "program assignment 2\\data_prog2\\"
sTest = "test_data\\"
sTrain = "train_data\\"

# image number
nImage = 120
sImage = str(nImage).zfill(5)
suffix = ".jpg"

# handle the image
img = Image.open(gUrl+lUrl+sTrain+sImage+suffix)
img.show()

