#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103,W0105,C0301

'''
Test for pickle
'''
import pickle as pkl
import tensorflow as tf
import numpy as np

# output weights
def fun_outWeight(weights):
    '''output weights as file'''
    # run session
    sessW = tf.Session()
    sWeights = sessW.run(weights)
    print(sWeights)
    filePointer = open("Pickletest.txt", "wb")
    pkl.dump(sWeights, filePointer)
    filePointer.close()
    return

# input weights
def fun_inWeight():
    '''input weights from file'''
    weights = pkl.load(open("Pickletest.txt", "rb"))
    print(weights)
    return weights

'''MAIN'''
weightn = np.mat('1,2;3,4;6,5')
weight = tf.constant(weightn, tf.int32)
sess = tf.Session()
weight1 = sess.run(weight)
print("weight1:", weight1)
fun_outWeight(weight)
weight2 = fun_inWeight()
print("weight2:", weight2)
