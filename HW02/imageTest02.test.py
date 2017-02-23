#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=I0011,C0103

'''
Just want to load image via python3.5
'''

import tensorflow as tf


url = "C:\\Users\\Yufei\\Desktop\\deep learning\\program assignment 2\\data_prog2\\train_data\\"
fileN = "00001.jpg"
file_contents = tf.read_file(url+fileN)
example = tf.image.decode_jpeg(file_contents, channels=3)
examplet = tf.transpose(example)
