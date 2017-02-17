#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103

'''
Filename: HW01c.py

'''

# import
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# prepare a session
sess = tf.Session()

# load data
file = open('HW01.txt')
data = [[float(x) for x in p.split()] for p in file]
# load x,y
x = tf.slice(data, [0, 0], [-1, 10])
x = tf.transpose(x)
x1 = tf.ones([1, 10000], tf.float32)
# add w0 terms
X = tf.concat(0, [x, x1])
X = tf.transpose(X)
Y = tf.slice(data, [0, 10], [-1, 1])

# Init iteration parameter
rate = 0.001
#itMax = 1000
itMax = 100
dataL = 10000
batchSize = 20
#batchSize = 100
theta = 0.01*tf.ones([11, 1], tf.float32)
# batch
X_batch = tf.reshape(X, [int(dataL/batchSize), batchSize, 10+1])
Y_batch = tf.reshape(Y, [int(dataL/batchSize), batchSize, 1])
X_batch = sess.run(X_batch)
Y_batch = sess.run(Y_batch)
# prepare for graph
X_axis = []
Y_axis = []
# iteration
for i in range(itMax):
    # random batch
    randSelect = random.randint(0, dataL/batchSize-1)
    Xb = X_batch[randSelect]
    Yb = Y_batch[randSelect]
    # calculate gradient
    op1 = tf.subtract(tf.matmul(Xb, theta), Yb)
    op2 = tf.matrix_transpose(Xb)
    gradient = 1/dataL * tf.matmul(op2, op1)
    # update theta
    theta = tf.subtract(theta, rate*gradient)
    # calculate cost
    cost = 0.5*1/dataL*tf.matmul(tf.transpose(op1), op1)
    print('iteration', i)
    # update axis
    X_axis.append(i+1)
    Y_axis.append(cost)
print('optimization finished')
# output / graph
Y_axis = tf.reshape(Y_axis, [itMax])
Y_axis = sess.run(Y_axis)
print('Weights:\n')
print(sess.run(tf.transpose(theta)))
# plot
plt.plot(X_axis, Y_axis, '-')
plt.xlabel('Iterations')
plt.ylabel('Err')
plt.show()
