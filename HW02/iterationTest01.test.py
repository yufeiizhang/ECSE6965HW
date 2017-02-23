#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103,W0105

'''
Program Assigment 2
1st question
'''

import tensorflow as tf
import numpy as np

a = np.mat('1,2;3,4;6,5')
A = tf.constant(a, tf.float32)

print(A)
B = A[:, 1]
sess = tf.Session()
b = sess.run(B)
print(b)

C = tf.argmax(A, 1)
c = sess.run(C)
print(c)

D = tf.exp(A)
d = sess.run(D)
print(d)

E = tf.reduce_sum(A, 1)
e = sess.run(E)
print(e)
