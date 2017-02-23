#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103,W0105,C0301

'''
Program Assigment 2
2nd question
Matrix version
Feb 22, 2017
'''

# import image module
import pickle as pkl
from PIL import Image
import tensorflow as tf

'''PRE DEFINED FUNCTIONS'''
# input weights
def fun_inWeight(file="PickleWeight.txt"):
    '''input weights from file'''
    weights = pkl.load(open(file, "rb"))
    print(weights)
    return weights

# load data from image dataset
def fun_loadImg(num, url="", digitNum=5):
    '''
    load data from image dataset
    num is the input image number
    url is the folder name of image dataset
    '''
    imgTrain = []
    suffix = ".jpg"
    # read individual image by loop
    for im in range(num):
        # construct string for image
        nImage = im+1
        sImage = str(nImage).zfill(digitNum)
        # open image file
        imgIm = Image.open(url+sImage+suffix)
        imgIm = imgIm.convert("L")
        # convert to list
        img1d = imgIm.getdata()
        # translate to list
        img1D = [i for i in img1d]
        # normalize
        img1N = [i/255.0 for i in img1D]
        # append to train dataset
        imgTrain.append(img1N)
        # append 1 to x vector
    imgTrain = tf.transpose(imgTrain)
    x1 = tf.ones([1, num], tf.float32)
    ResultImgSet = tf.concat(0, [imgTrain, x1])
    ResultImgSet = tf.transpose(ResultImgSet)
    return ResultImgSet

# load data from label dataset
def fun_loadLabel(num, url="train_label.txt"):
    '''
    num is the input image number
    url is the folder name of label dataset, default local training set
    '''
    # load data from train label set
    file = open(url)
    data = [[int(x) for x in p.split()] for p in file]
    # reload
    rData = [data[i] for i in range(num)]
    # get rid of []s
    rData = sum(rData, [])
    # load in tensorflow
    ResultLabSet = tf.constant(rData, tf.int32)
    # rData-1 here for onehot
    subOne = tf.ones([num], tf.int32)
    ResultLabel4 = tf.subtract(ResultLabSet, subOne)
    # to onehot
    ResultLabel = tf.one_hot(ResultLabel4, 5, on_value=1, off_value=0)
    ResultLabelf = tf.to_float(ResultLabel)
    return ResultLabelf

'''MAIN'''
# reload weights
w = fun_inWeight()
weight = tf.constant(w, tf.float32)
# session
sess = tf.Session()

'''TEST RESULT'''
#nImgTest = 4982
nImgTest = 4980
# prepare urls
gUrl = "C:\\Users\\Yufei\\Desktop\\deep learning\\"
lUrl = "program assignment 2\\data_prog2\\"
sTest = "test_data\\"
sLabel = "labels\\"
slTest = "test_label.txt"
# url for Image Testing dataset
imgtUrl = gUrl+lUrl+sTest
labtUrl = gUrl+lUrl+sLabel+slTest
# load data from testing dataset
ImgTest = fun_loadImg(nImgTest, imgtUrl, 4)
# load label from training dataset
lTest = fun_loadLabel(nImgTest, labtUrl)
print("Finish Load Test Data")
# run classification
XW = tf.matmul(ImgTest, weight)
indXW = tf.argmax(XW, 1)
# run label
indLT = tf.argmax(lTest, 1)
# True ratio
print("Finish Calculate Training Data")
# check the result
subXW = tf.subtract(indXW, indLT)
wrongNum = tf.count_nonzero(subXW)
wrongOut = sess.run(wrongNum)
# show the result
TRatio = (nImgTest-wrongOut)/(nImgTest)
print("True ratio: ", TRatio)



