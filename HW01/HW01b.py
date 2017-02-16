#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103

'''
Filename: HW01b.py
'''

# import
import tensorflow as tf
import matplotlib.pyplot as plt

# prepare a session
sess = tf.Session()

# load data
file = open('HW01.txt')
data = [[float(x) for x in p.split()] for p in file]
# load x,y
X = tf.slice(data, [0, 0], [-1, 10])
Y = tf.slice(data, [0, 10], [-1, 1])

# Init iteration parameter
rate = 0.001
itMax = 1000
#itMax = 10
dataL = 10000
theta = 0.01*tf.ones([10, 1], tf.float32)
# prepare for graph
X_axis = []
Y_axis = []
# iteration
for i in range(itMax):
    # calculate gradient
    op1 = tf.subtract(tf.matmul(X, theta), Y)
    op2 = tf.matrix_transpose(X)
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