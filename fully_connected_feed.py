# -*- coding=utf-8 -*-
import argparse
import os
import sys
import time
import numpy as np
import pickle as pk
from jobman import DD
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import random
import mnist

FLAGS = None

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32)
  labels_placeholder = tf.placeholder(tf.int32)
  return images_placeholder, labels_placeholder

def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches SGD
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = np.split(np.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = np.split(np.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [np.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx

def fill_feed_dict(data_set, images_pl, labels_pl):
  #create feed dict with next batch
  #idx is the original index of dataset
  #counter is to count batches in one epoch
  #index is index of dataset after shuffling
  idx=data_set.idx
  counter=data_set.counter
  if counter==data_set.steps_each_epoch or counter==0:
    counter=0
    index = range(data_set.num_examples)
    random.shuffle(index)
    data_set['index']=index
  else:
    index=data_set.index
  batch_index=[index[i] for i in idx[counter]]
  images_feed=[data_set.fea[i] for i in batch_index]
  labels_feed=[data_set.label[i] for i in batch_index]
  
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  counter=counter+1
  data_set.counter=counter
  return feed_dict

def fill_feed_dict_for_do_eval(data_set, images_pl, labels_pl):
  #create feed dict with next batch
  #idx is the original index of dataset
  #counter is to count batches in one epoch
  #index is index of dataset after shuffling
  idx=data_set.idx
  counter=data_set.counter
  if counter==data_set.steps_each_epoch or counter==0:
    counter=0
    index = range(data_set.num_examples)
    random.shuffle(index)
    data_set['index']=index
  else:
    index=data_set.index
  batch_index=[index[i] for i in idx[counter]]
  images_feed=[data_set.fea[i] for i in batch_index]
  labels_feed=[data_set.label[i] for i in batch_index]
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  counter=counter+1
  data_set.counter=counter
  return feed_dict,labels_feed


def do_eval(sess,
            eval_correct,h_,
            images_placeholder,
            labels_placeholder,
            data_set):
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  fully_connect_output=[]
  label_fulcon=[]
  data_set.counter=0#init from scrach
  for step in range(data_set.steps_each_epoch):
    feed_dict,labels_feed = fill_feed_dict_for_do_eval(data_set,
                               images_placeholder,
                               labels_placeholder)
    eval_correct_value,h_value = sess.run([eval_correct,h_], feed_dict=feed_dict)
    true_count += eval_correct_value
    fully_connect_output.extend(h_value.tolist())
    label_fulcon.extend(labels_feed)
  precision = float(true_count) / data_set.num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (data_set.num_examples, true_count, precision))
  return fully_connect_output,label_fulcon#need record the output from fully connected and its label for svm classification


def run_training():
  dataPath=FLAGS.input_data_dir
  data_sets=DD()
  with open(dataPath,'rb') as f:
    MSR3D=pk.load(f)
  data_sets.train=DD({ 'fea':MSR3D["train_fea"],'label':MSR3D["train_label"],'num_examples':len(MSR3D["train_label"]) })
  data_sets.validation=DD({ 'fea':MSR3D["val_fea"],'label':MSR3D["val_label"],'num_examples':len(MSR3D["val_label"]) })
  data_sets.test=DD({ 'fea':MSR3D["test_fea"],'label':MSR3D["test_label"],'num_examples':len(MSR3D["test_label"]) })
  
  idx=generate_minibatch_idx(data_sets.train.num_examples,FLAGS.batch_size)
  data_sets.train.idx=idx
  data_sets.train.steps_each_epoch=len(idx)
  data_sets.train.counter=0
  print "train counter init!"
  idx=generate_minibatch_idx(data_sets.validation.num_examples,FLAGS.batch_size)
  data_sets.validation.idx=idx
  data_sets.validation.steps_each_epoch=len(idx)
  data_sets.validation.counter=0
  print "validation counter init!"
  idx=generate_minibatch_idx(data_sets.test.num_examples,FLAGS.batch_size)
  data_sets.test.idx=idx
  data_sets.test.steps_each_epoch=len(idx)
  data_sets.test.counter=0
  print "test counter init!"
  
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits,dense,dense_biases,h_,weights= mnist.inference(images_placeholder,
                             FLAGS.hidden1)
    #add l2-regularization to loss
    weights_list=[dense,weights]
    loss = mnist.loss(logits, labels_placeholder,weights_list)
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    #saver = tf.train.Saver()
    #saver = tf.train.Saver({"dense":dense,"dense_biases":dense_biases})

    # Create a session for running Ops on the Graph.
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

    print('Training Data Eval:')
    train_output,train_label=do_eval(sess,
            eval_correct,h_,
            images_placeholder,
            labels_placeholder,
            data_sets.train)
    # Evaluate the validation set.
    print('Validation Data Eval:')
    val_output,val_label=do_eval(sess,
            eval_correct,h_,
            images_placeholder,
            labels_placeholder,
            data_sets.validation)
    # Evaluate the test set.
    print('Test Data Eval:')
    test_output,test_label=do_eval(sess,
            eval_correct,h_,
            images_placeholder,
            labels_placeholder,
            data_sets.test)
        

    
    #save
    MSR3D_fulcon=DD()
    MSR3D_fulcon.train=DD()
    MSR3D_fulcon.val=DD()
    MSR3D_fulcon.test=DD()
    MSR3D_fulcon.train.fea=train_output
    MSR3D_fulcon.train.label=train_label
    MSR3D_fulcon.val.fea=val_output
    MSR3D_fulcon.val.label=val_label
    MSR3D_fulcon.test.fea=test_output
    MSR3D_fulcon.test.label=test_label
    with open("./MSR3D_fulcon.pkl",'wb') as f:
      pk.dump(MSR3D_fulcon,f)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      #default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
      #                     'tensorflow/mnist/input_data'),
      default="./feature_label_MSR3D_lower_triangular.pkl",
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      #default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
      #                    'tensorflow/mnist/logs/fully_connected_feed'),
      default='./logs',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
