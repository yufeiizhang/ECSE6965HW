#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103,W0105,C0301

'''
Program Assigment 2
1st question
Matrix version
Use placeholder
For higher Order
(edit version of 02a.8)
Feb 22, 2017
'''

# import image module
import random
import pickle as pkl
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

'''PRE DEFINED FUNCTIONS'''
# plot weights
def fun_pltWeights(weights, dim, save=False):
    '''
    As defined in question,
    input weights is a tensor, (5,28^2+1)
    and dim is the dimention to plt, from 0-4
    '''
    print("Plot Weight: ", dim)
    '''Use non-tensorflow method instead'''
    # cut
    weightD = weights[0:28**2, dim]
    # reshape
    weightC = weightD.reshape([28, 28])
    # plt
    plt.imshow(weightC)
    plt.colorbar()
    if save:
        # image name
        sImage = str(dim).zfill(2)
        sImg = sImage+".jpg"
        plt.savefig(sImg)
    else:
        plt.show()
    plt.close()
    return

# output weights
def fun_outWeight(weights, name="PickleWeight.txt"):
    '''output weights as file'''
    # run session
    #sessW = tf.Session()
    #sWeights = sessW.run(weights)
    print(weights)
    filePointer = open(name, "wb")
    pkl.dump(weights, filePointer)
    filePointer.close()
    return

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

# batch Generator
def fun_genBatch(batSize, dataL, X, Y, imgShape):
    '''random generate batch'''
    # generate an unrepeated list
    sel = random.sample(range(dataL), batSize)
    # empty list
    Xb = [X[i] for i in sel]
    Yb = [Y[i] for i in sel]
    # reshape batch
    Yb2 = tf.reshape(Yb, [_batchSize_, _classNum_])
    Xb2 = tf.reshape(Xb, [_batchSize_, imgShape])
    return Xb2, Yb2

def fun_thetaUpdate(weights, Xb, Yb, pEta, pLambda):
    '''Update weights for k-class'''
    # gradient terms
    # loss gradient
    #dWeight = fun_gradLoss(weights, Xb, Yb, batSize, classNum)
    XTY = tf.matmul(Xb, Yb, transpose_a=True)
    # calculate exp(theta_k X[m]) for everyone
    expXW = tf.exp(tf.matmul(Xb, weights))
    # sum for k class
    SumExpXW = tf.reduce_sum(expXW, 1, keep_dims=True)
    # calculate the fraction
    XEXP = tf.matmul(Xb, tf.divide(expXW, SumExpXW), transpose_a=True)
    # calculate gradient for weight
    dWeight = tf.subtract(XEXP, XTY)
    # regulation terms
    #dL2Norm = fun_gradL2Norm(weights, pLambda)
    dL2Norm = tf.scalar_mul(2*pLambda, weights)
    # sum the gradient &
    # multiply hyper-parameter eta
    op2 = tf.scalar_mul(-pEta, tf.add(dL2Norm, dWeight))
    # update weights
    weightNew = tf.add(weights, op2)
    return weightNew

'''MAIN'''
print("Finish Import")
# session for all
sess = tf.Session()

# prepare urls
gUrl = "C:\\Users\\Yufei\\Desktop\\deep learning\\"
lUrl = "program assignment 2\\data_prog2\\"
sTrain = "train_data\\"
sLabel = "labels\\"
slTrain = "train_label.txt"
# url for Image Training dataset
imgTUrl = gUrl+lUrl+sTrain
labTUrl = gUrl+lUrl+sLabel+slTrain

'''IMPORTANT PARAMETERS'''
#nImgTrain = 25112
nImgTrain = 25110
_classNum_ = 5
# iteration times and batch
_itMax_ = 50
_OuterLoop_ = 4
_batchSize_ = 128
_shape_ = 28**2+1
# hyper parameter
_Eta_ = 0.0001
_Lambda_ = 0.00002

'''LOAD TRAIN DATA'''
# load data from training dataset
tfImgTrain = fun_loadImg(nImgTrain, imgTUrl)
# load label from training dataset
tflTrain = fun_loadLabel(nImgTrain, labTUrl)
# generate weight
tfWeight = tf.ones([_shape_, _classNum_], tf.float32)
tfWeight = tf.mul(0.01, tfWeight)

# trans back to normal
tftrans = tf.Session()
imgtrain = tftrans.run(tfImgTrain)
ltrain = tftrans.run(tflTrain)
nweight = tftrans.run(tfWeight)
print("Finish Load Train Data")


'''ITERATION'''
for outer in range(_OuterLoop_):
    _X_ = tf.placeholder(tf.float32, shape=(nImgTrain, _shape_))
    _Y_ = tf.placeholder(tf.float32, shape=(nImgTrain, _classNum_))
    _W_ = tf.placeholder(tf.float32, shape=(_shape_, _classNum_))
    weight = _W_
    for index in range(_itMax_):
        # output every iteration
        print("Iteration: ", index+1)
        # get the batch
        imgBatch, labBatch = fun_genBatch(_batchSize_, nImgTrain, _X_, _Y_, _shape_)
        # Update weights for different class
        #weightN = fun_thetaUpdate(weight, imgBatch, labBatch, _Eta_, _Lambda_)
        weight = fun_thetaUpdate(weight, imgBatch, labBatch, _Eta_, _Lambda_)
        # reshape the results (just ensure)
        #weight = tf.reshape(weightN, [_shape_, _classNum_])
    weightout = weight

    '''OUTPUT'''
    # load from previous weight
    if outer == -1:
        # outer == 0 previously
        nweight = fun_inWeight("PreWeight.txt")
    else:
        nweight = fun_inWeight("temp.txt")
    # output weights
    print("RUN Session: ", outer+1)
    sWeight = sess.run(weightout, feed_dict={_X_:imgtrain, _Y_:ltrain, _W_:nweight})
    print("FINISH")
    fun_outWeight(sWeight, "temp.txt")
fun_outWeight(sWeight)
# reload weights
w = fun_inWeight()
# plot weights
for index in range(_classNum_):
    fun_pltWeights(w, index, True)

