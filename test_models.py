## verify.py -- check the accuracy of a neural network
##
## Copyright (C) IBM Corp, 2017-2018
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_cifar import CIFAR, CIFARModel
from setup_mnist_noise import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "imagenet"], default="mnist")
parser.add_argument("-D", "--defense", action='store_true', help="activate the defensive noise strategy")
parser.add_argument("-M", "--mixture", action='store_true', help="activate the defensive noise mixture strategy")
parser.add_argument("-S", "--shuffle", action='store_true', help="activate the shuffling defense strategy")
parser.add_argument("-b", "--batchsize", type=int, default=100)
parser.add_argument("-r", "--runs", type=int, default=1)
parser.add_argument("-n", "--noise", type=float, default=1.0)
args = vars(parser.parse_args())

BATCH_SIZE = args['batchsize']

with tf.Session() as sess:
    if args['dataset'] == 'mnist':
        data, model = MNIST(), MNISTModel(restore="models/mnist", session=sess, use_log=True,noise=0.01)
    elif args['dataset'] == 'cifar10':
        data, model = CIFAR(), CIFARModel("models/cifar-distilled-100", sess)
    elif args['dataset'] == 'imagenet':
        data, model = ImageNet(args['batchsize']), InceptionModel(sess)
       
    for _ in range(args['runs']): 
        x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
        y = model.predict(x)
        if args['defense']:
            y = y+tf.keras.backend.random_normal(tf.shape(y), mean=0.0, stddev=args['noise'])
        elif args['shuffle']:
            # TODO: implement non-maximal class swapping
            # get the index, probability of the highest confidence class
            # randomly shuffle all the other classes
            # return the new vector as y
            pass
        elif args['mixture']:   
            std_params =  [0.1, 0.05, 0.03, 0.02, 0.01]
            mean_params = [0, 0.5, 0.75, 0.25, 1.0]
            M = len(std_params)
            def gmm_noise(p):
                deltas = tf.reduce_max(p) - p
                delta2 = tf.reduce_max(deltas)
                eps_vecs = [tf.keras.backend.random_normal(tf.shape(p), 
                            mean=delta2*mean_params[i],
                            stddev=std_params[i]) for i in range(M)]                
                m = tf.random.uniform((1,), minval=0, maxval=M, dtype=tf.int32)
                return tf.nn.embedding_lookup(tf.stack(eps_vecs), m)
            noise = tf.squeeze(tf.map_fn(gmm_noise, y))
            y = y+noise
        r = []
        for i in range(0,len(data.test_data),BATCH_SIZE):
            pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})            
            #print(pred)
            #print('real',data.test_labels[i],'pred',np.argmax(pred))
            r.append(np.argmax(pred,1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
        print(np.mean(r))
