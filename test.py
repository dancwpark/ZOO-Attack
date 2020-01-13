from numpy.random import seed
#seed(1)
import tensorflow as tf
#tf.set_random_seed(0)
import os
import sys
import numpy as np
import random
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_yaml
from PIL import Image

from setup_simple_mnist import SMNIST, SMNISTModel

def train_target(data, file_name, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = SMNISTModel(use_log=True).model
    if init != None:
        model.load_weights(init)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])

    model.fit(data.train_data, data.train_labels,
            batch_size=batch_size,
            validation_data=(data.validation_data, data.validation_labels),
            nb_epoch=num_epochs,
            shuffle=False)

    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    print("Target model test accuracy on clean data: ", score[1])
    return model

def main():
    print()
    print("+---------------------+")
    print("|Training Target Model|")
    print("+---------------------+")
    train_target(SMNIST(), "", num_epochs=5)

if __name__ == "__main__":
    main()

