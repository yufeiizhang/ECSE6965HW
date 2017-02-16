#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103

'''
Filename: HW01a.py

'''

# import
import tensorflow as tf

# prepare a session
sess = tf.Session()

# load data
file = open('HW01.txt')
data = [[float(x) for x in p.split()] for p in file]
# load x,y
X = tf.slice(data, [0, 0], [-1, 10])
Y = tf.slice(data, [0, 10], [-1, 1])
# calculate
op1 = tf.matmul(tf.matrix_transpose(X), X)
op2 = tf.matrix_inverse(op1)
op3 = tf.matmul(tf.transpose(X), Y)
theta = tf.matmul(op2, op3)
print('theta:\n', sess.run(theta))
