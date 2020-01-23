"""
This code is testing the effectiveness of substitute black box attack assuming
the output from the black box target model is ONLY the resulting label.

The substitute model is trained on the test set with the target's label
predictions.

CW white box attack is used on the substitute model. 

The successful attacks are also tested against the black box target model.
"""
import os
import sys
import tensorflow as tf
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
#from l2_attack import CarliniL2
from l2_attack_w_defense import CarliniL2 # with defense and averaging
################
# Target Model #
################
from setup_mnist import MNIST, MNISTModel
####################
# Substitute Model #
####################
from setup_mnist_sub import MNIST_sub, MNISTModel_sub

def train_target(data, file_name, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = MNISTModel(use_log=True).model
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
            shuffle=True)
    if file_name != None:
        model.save(file_name)

    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    print("Target model test accuracy on clean data: ", score[1])
    return model

def get_target_labels(data):
    """
    Get labels as returned by target model.
    Just the label, no prob
    """
    print("Getting labels from target")
    model = MNISTModel().model
    model.load_weights("models/mnist")
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])
    train_labels = model.predict(data.train_data)
    validation_labels = model.predict(data.validation_data)
    test_labels = model.predict(data.test_data)
    #  Need to turn [p, p, ..., p] to y
    train_labels = train_labels.argmax(axis=-1)
    validation_labels = validation_labels.argmax(axis=-1)
    test_labels = test_labels.argmax(axis=-1)
    # Turn y to [0, 0, ..., 1]
    train_labels = keras.utils.to_categorical(train_labels, 10)
    validation_labels = keras.utils.to_categorical(validation_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)

    return (train_labels, validation_labels, test_labels)

def train_substitute(data, file_name, tl, num_epochs=1, batch_size=128, train_temp=1, init=None):
    """
    Create a model (different than target) and train using true data
    samples and labels from the target model)
    """
    # Get labels for training and validation data
    test_labels = tl
    
    # Create another model
    model = Sequential()
    # input shape ordering
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (1, 28, 28)
    else:
        input_shape = (28, 28, 1)

    layers = [Flatten(input_shape=input_shape),
            Dense(200),
            Activation('relu'),
            Dropout(0.5),
            Dense(200),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')]
    for layer in layers:
        model.add(layer)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])
    model.fit(data.test_data[:5000], test_labels[:5000],
        batch_size=batch_size,
        validation_data=(data.test_data[5000:], test_labels[5000:]),
        nb_epoch=num_epochs,
        shuffle=True)
    if file_name != None:
        model.save(file_name)

    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    print("Substitute model test accuracy on clean data: ", score[1])

def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets instead of 1000
    """
    inputs = []
    targets = []
    labels = []
    true_ids = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])
            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                labels.append(data.test_labels[start+i])
                true_ids.append(start+i)
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])
            labels.append(data.test_labels[start+i])
            true_ids.append(start+i)

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    true_ids = np.array(true_ids)

    return inputs, targets, labels, true_ids

def show(img, name='output.png', save=False):
    """
    Show MNIST digits in console.
    """
    if save:
        np.save(name, img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    if save:
        pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def attack_substitute(data, subfile, targetfile, stop=True):
    with tf.Session() as sess:
        print(">> Loading Models")
        model = MNISTModel_sub(restore=subfile, session=sess, use_log=True)
        model_target = MNISTModel(restore=targetfile, session=sess, use_log=True)
        print("... Done ...")

        # Attack
        use_log = True
        random.seed(1216)
        np.random.seed(1216)

        attack = CarliniL2(sess, 
                model, 
                batch_size=1, 
                max_iterations=3000, 
                print_every=100, 
                confidence=0, 
                use_log=use_log, 
                early_stop_iters=100,
                learning_rate=1e-2,
                initial_const=.5,
                binary_search_steps=1,
                targeted = True,
                defense=True) # noise choose

        all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, 
                samples=20, 
                targeted=True, 
                start=1, 
                inception=False)
        
        img_no = 0
        total_success = 0
        total_transfers = 0
        l2_total = 0.
        for i in range(all_true_ids.size):
            inputs = all_inputs[i:i+1]
            targets = all_targets[i:i+1]
            labels = all_labels[i:i+1]
            print("true labels:", np.argmax(labels), labels)
            print("target:", np.argmax(targets), targets)
            # test if the image is correctly classified
            original_predict = model.model.predict(inputs)
            original_predict = np.squeeze(original_predict)
            original_prob = np.sort(original_predict)
            original_class = np.argsort(original_predict)
            print("original probs:", original_prob[-1:-6:-1])
            print("original class:", original_class[-1:-6:-1])
            print("original probs (most unlikely):", original_prob[:6])
            print("original class (most unliekly):", original_class[:6])
            if original_class[-1] != np.argmax(labels):
                print("Skip: wrongly classified image no. {}, original class {}, classified as {}".format(i, np.argmax(labels), original_class[-1]))
                continue

            img_no += 1
            timestart = time.time()
            #adv, const = attack.attack_batch(inputs, targets)
            adv, const, q_count = attack.attack_batch(inputs, targets) # w/ defense
            if type(const) is list:
                const = const[0]
            if len(adv.shape) == 3:
                adv = adv.reshape((1,) + adv.shape)
            timeend = time.time()
            l2_distortion = np.sum((adv-inputs)**2)**.5
            adversarial_predict = model.model.predict(adv)
            adversarial_predict = np.squeeze(adversarial_predict)
            adversarial_prob = np.sort(adversarial_predict)
            adversarial_class = np.argsort(adversarial_predict)
            print("adversarial probabilities:", adversarial_prob[-1:-6:-1])
            print("adversarial classification:", adversarial_class[-1:-6:-1])
            success = False
            if adversarial_class[-1] == np.argmax(targets):
                success = True
            if l2_distortion > 20.:
                success = False
            if success:
                print("+-------+")
                print("|SUCCESS|")
                print("+-------+")
                total_success += 1
                l2_total += l2_distortion
                print("original")
                show(inputs)
                print("adversarial")
                show(adv)
                print("diff")
                show(adv - inputs)
                print("Test on target model")
                target_prediction = model_target.model.predict(adv)
                target_prediction = np.squeeze(target_prediction)
                target_class = np.argsort(target_prediction)
                if target_class[-1] != original_class[-1]:
                    print(">> :)  Adversarial example transfers")
                    total_transfers += 1
                else:
                    print(">> :( Does not transfer")
                if stop:
                    input(">>> Enter to continue")
            else:
                print("+----+")
                print("|FAIL|")
                print("+----+")
            
            print("[STATS][L1] total={}, seq={}, id={}, time={:.3f}, success={}, const={:.6f}, prev_class={}, new_class={}, distortion={:.5f}, success_rate={:.3f}, l2_avg={:.5f}, transfer_rate={}".format(img_no, i, all_true_ids[i], timeend-timestart, success, const, original_class[-1], adversarial_class[-1], l2_distortion, total_success/float(img_no), 0 if total_success==0 else l2_total/total_success, 0 if total_success==0 else total_transfers/total_success))
            print()
            print()
            print()
            sys.stdout.flush()
        
        print("Done with experiments")
        print("Stats of interest")
        print("Successes:", total_success)
        print("Transfers:", total_transfers)
        print("Transfer Rate: {}".format(0 if total_success==0 else total_transfers/total_success))


def main():
    print()
    print("+---------------------+")
    print("|Training Target Model|")
    print("+---------------------+")
    train_target(MNIST(), "models/mnist", num_epochs=5)
    print("+------------------+")
    print("|Training Sub Model|")
    print("+------------------+")
    training_labels, validation_labels, test_labels = get_target_labels(MNIST())
    train_substitute(MNIST(), "models/sub", test_labels, num_epochs=5)
    print("+------+")
    print("|Attack|")
    print("+------+")
    m = MNIST()
    m.train_labels = training_labels
    m.validation_labels = validation_labels
    m.test_labels = test_labels
    attack_substitute(m, "models/sub", "models/mnist", stop=False)

if __name__ == "__main__":
    save_location = "models"
    os.system("mkdir -p {}".format(save_location))
    main()

