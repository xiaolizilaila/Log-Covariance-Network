# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import math
import tensorflow as tf

NUM_CLASSES = 20
IMAGE_PIXELS = 1830 #the length of the feature


#def inference(images, hidden1_units, hidden2_units):
def inference(images, hidden1_units):
  # Hidden 1
  with tf.name_scope('hidden1'):
    dense = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='dense')
    dense_biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='dense_biases')
    hidden1 = tf.nn.relu(tf.matmul(images, dense) + dense_biases,name='hidden1_op')
  #linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden1, weights) + biases
  return logits,dense,dense_biases,hidden1,weights


def loss(logits, labels,weights_list):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  regularizer=tf.contrib.layers.l2_regularizer(0.01)
  summed_penalty=tf.contrib.layers.apply_regularization(regularizer, weights_list=weights_list)
  #print summed_penalty
  return tf.reduce_mean(cross_entropy+summed_penalty, name='xentropy_mean')


def training(loss, learning_rate):
  #tf.summary.scalar('loss', loss)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
