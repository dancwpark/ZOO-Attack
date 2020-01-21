"""
This code is testing the following:

Test the effectiveness of the substitution attack aginst
Complex vs Simple target architectures
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
from PIL import Image
from l2_attack import CarliniL2
################
# Target Model #
################
from setup_mnist import MNIST, MNISTModel
######################################
# Simple model for transference test #
######################################
# SMNISTModel_'n' -- higher 'n' means simpler
#from setup_mnist import SMNISTModel_1 as SMNISTModel
#from setup_mnist import SMNISTModel_2 as SMNISTModel
from setup_mnist import SMNISTModel_3 as SMNISTModel
from setup_mnist import SMNISTModel_4 as SMNISTModel
####################
# Substitute Model #
####################
# MNISTModel_sub'n' -- higher 'n' means more complex
# Alose has one example of just using the actual same arch
from setup_mnist_sub import MNISTModel_sub
#from setup_mnist import MNISTModel as MNISTModel_sub
#from setup_mnist_sub import MNISTModel_sub2 as MNISTModel_sub
#from setup_mnist_sub import MNISTModel_sub3 as MNISTModel_sub
#from setup_mnist_sub import MNISTModel_sub4 as MNISTModel_sub


####################
# Global Variables #
####################
target_a = 0. # Target Model's accuracy on half of test set
simple_a = 0. # Simple Model's accuracy on half of test set
subsimple_a = 0. # Substitute trained on simple model acc.
subtarget_a = 0. # Substitute trained on target model acc.

# The training functions are separate for 
## separated access to global variables
def train_target(data, file_name, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    global target_a
    model = MNISTModel(use_log=True).model
    if init != None:
        model.load_weights(init)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])
    # Train model
    model.fit(data.train_data, data.train_labels,
            batch_size=batch_size,
            validation_data=(data.validation_data, data.validation_labels),
            nb_epoch=num_epochs,
            shuffle=False)
    # Save model if location given
    if file_name != None:
        model.save(file_name)

    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    target_a = score[1]
    print("Target model test accuracy on clean data: ", score[1])
    return model

def train_simple(data, file_name, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    global simple_a
    model = SMNISTModel(use_log=True).model
    if init != None:
        model.load_weights(init)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])
    # Train model
    model.fit(data.train_data, data.train_labels,
            batch_size=batch_size,
            validation_data=(data.validation_data, data.validation_labels),
            nb_epoch=num_epochs,
            shuffle=False)
    # Save if location given
    if file_name != None:
        model.save(file_name)

    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    simple_a = score[1]
    print("Simple model test accuracy on clean data: ", score[1])
    return model

# TODO start:
## get_target_labels and get_target_labels_simple can
## be combined into one
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

def get_target_labels_simple(data):
    """
    Get labels as returned by model with loaded 
    weights.
    Just the label, no prob
    data = MNIST() for example
    model = MNISTModel() for example
    load =  "models/mnist" for example
    """
    print("Getting labels from specified model")
    s_model = SMNISTModel().model
    s_model.load_weights("models/simple")
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    s_model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])
    # Get class probabilities
    train_labels = s_model.predict(data.train_data)
    validation_labels = s_model.predict(data.validation_data)
    test_labels = s_model.predict(data.test_data)
    # Get class prediction
    train_labels = train_labels.argmax(axis=-1)
    validation_labels = validation_labels.argmax(axis=-1)
    test_labels = test_labels.argmax(axis=-1)
    # Get binary vector
    train_labels = keras.utils.to_categorical(train_labels, 10)
    validation_labels = keras.utils.to_categorical(validation_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)

    return (train_labels, validation_labels, test_labels)
# TODO end


# Not currently used
def train_substitute_val(data, file_name, num_epochs=1, batch_size=128, train_temp=1, init=None):
    """
    Use the validation and test set to train the substitute model
    """
    global sub_a
    # Get labels for training and validation data
    test_labels = data.test_labels
    
    model = MNISTModel_sub(use_log=True).model
    if init != None:
        model.load_weights(init)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])
 
    model.fit(data.test_data[:5000], test_labels[:5000],
        batch_size=batch_size,
        validation_data=(data.test_data[5000:], test_labels[5000:]),
        nb_epoch=num_epochs,
        shuffle=True)
    
    model.fit(data.validation_data, data.validation_labels,
            batch_size=batch_size,
            validation_data=(data.test_data[5000:], test_labels[5000:]),
            nb_epoch=num_epochs,
            shuffle=True)

    if file_name != None:
        model.save(file_name)

    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    sub_a = score[1]
    print("Substitute model test accuracy on clean data: ", score[1])


def train_substitute(data, file_name, num_epochs=1, batch_size=128, train_temp=1, init=None):
    """
    Create a model (different than target) and train using true data
    samples and labels from the target model)
    """
    # Get labels
    test_labels = data.test_labels
    # Create the substitute model
    model = MNISTModel_sub(use_log=True).model
    if init != None:
        model.load_weights(init)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'])
    # Train the substitute model
    model.fit(data.test_data[:5000], test_labels[:5000],
        batch_size=batch_size,
        validation_data=(data.test_data[5000:], test_labels[5000:]),
        nb_epoch=num_epochs,
        shuffle=True)
    # Save the substitute model for use later (in attack)
    if file_name != None:
        model.save(file_name)

    score = model.evaluate(data.test_data, data.test_labels, verbose=0)
    print("Substitute model test accuracy on clean data: ", score[1])
    return score[1]



def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    -- Taken from Carlini Attack code --
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


def attack_substitute(data, subfile, model_type, targetfile, stop=True):
    with tf.Session() as sess:
        print(">> Loading Models")
        # Load the trained substitute
        model = MNISTModel_sub(restore=subfile, session=sess, use_log=True)
        # Load model to be attacked
        model_target = model_type(restore=targetfile, session=sess, use_log=True).model
        
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
                targeted = True)

        all_inputs, all_targets, all_labels, all_true_ids = generate_data(data, 
                samples=20, 
                targeted=True, 
                start=1, 
                inception=False)
        
        img_no = 0
        total_success = 0 # Success on the substitute model
        total_transfers = 0 # Success on the target model
        l2_total = 0.
        for i in range(all_true_ids.size):
            inputs = all_inputs[i:i+1]
            targets = all_targets[i:i+1]
            labels = all_labels[i:i+1]
            print("true labels:", np.argmax(labels), labels)
            print("target:", np.argmax(targets), targets)
            # test if the image is correctly classified by the substitute
            # Get the substitute model's predicted label of the input
            original_predict = model.model.predict(inputs)
            original_predict = np.squeeze(original_predict)
            original_prob = np.sort(original_predict)
            original_class = np.argsort(original_predict)
            print("original probs:", original_prob[-1:-6:-1])
            print("original class:", original_class[-1:-6:-1])
            print("original probs (most unlikely):", original_prob[:6])
            print("original class (most unliekly):", original_class[:6])
            # If the predicted label is not the true labe, skip
            if original_class[-1] != np.argmax(labels):
                print("Skip: wrongly classified image no. {}, "
                      "original class {}, "
                      "classified as {}".format(i, 
                          np.argmax(labels), 
                          original_class[-1]))
                continue
            # Create the adversarial example
            img_no += 1
            timestart = time.time()
            adv, const = attack.attack_batch(inputs, targets) # HERE
            if type(const) is list:
                const = const[0]
            if len(adv.shape) == 3:
                adv = adv.reshape((1,) + adv.shape)
            timeend = time.time()
            l2_distortion = np.sum((adv-inputs)**2)**.5
            # Get the substitute model's prediction
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
                # See if adversarial example transfers to target
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
            
            print("[STATS][L1] total={}, seq={}, id={}, "
                  "time={:.3f}, success={}, const={:.6f}, "
                  "prev_class={}, new_class={}, distortion={:.5f}, "
                  "success_rate={:.3f}, l2_avg={:.5f}, "
                  "transfer_rate={}".format(img_no, i, all_true_ids[i], 
                      timeend-timestart, success, const, original_class[-1], 
                      adversarial_class[-1], l2_distortion, 
                      total_success/float(img_no), 
                      0 if total_success==0 else l2_total/total_success, 
                      0 if total_success==0 else total_transfers/total_success))
            print("\n\n\n")
            sys.stdout.flush()
        
        print("Done with experiments")
        # Print general stats of interest
        print("Target accuracy:", target_a)
        if model_type is MNISTModel:
            print("Substitute accuracy:", subtarget_a)
        else:
            print("Substitute accuracy:", subsimple_a)
        print("Stats of interest")
        print("Successes:", total_success)
        print("Transfers:", total_transfers)
        print("Transfer Rate: {}".format(0 if total_success==0 else total_transfers/total_success))
        return total_success, total_transfers

def main():
    global subtarget_a
    global subsimple_a
    print()
    print("+---------------------+")
    print("|Training Target Model|")
    print("+---------------------+")
    train_target(MNIST(), "models/mnist", num_epochs=5)
    print("+---------------------+")
    print("|Training Simple Model|")
    print("+---------------------+")
    train_simple(MNIST(), "models/simple", num_epochs=5)
    print("+------------------+")
    print("|Training Sub Model|")
    print("+------------------+")
    # Get labels from target model
    training_labels, validation_labels, test_labels = get_target_labels(MNIST())
    n = MNIST()
    n.train_labels = training_labels
    n.validation_labels = validation_labels
    n.test_labels = test_labels
    # Get labels from simple model
    training_labels, validation_labels, test_labels = get_target_labels_simple(MNIST())
    m = MNIST()
    m.train_labels = training_labels
    m.validation_labels = validation_labels
    m.test_labels = test_labels
    # Train two sub models using n and m (our new labels)
    subtarget_a = train_substitute(n, "models/sub_target", num_epochs=5)
    subsimple_a = train_substitute(m, "models/sub_simple", num_epochs=5)
    print("+------+")
    print("|Attack|")
    print("+------+")
    # Attack substitute trained on target
    t_succ, t_tran = attack_substitute(n, 
            "models/sub_target", 
            MNISTModel, 
            "models/mnist", 
            stop=False)
    #input(">>")
    # Attack substitute trained on simple
    s_succ, s_tran = attack_substitute(m, 
            "models/sub_simple", 
            SMNISTModel, 
            "models/simple", 
            stop=False)
    print("----------------------------------")
    print("STATS")
    print("Target acc: {}, \n"
          "sub_t acc: {},  \n"
          "Success on sub_t: {}, \n"
          "Transferred ex: {}, \n"
          "Transfer rate: {}".format(target_a, 
              subtarget_a, 
              t_succ, 
              t_tran,
              0 if t_succ==0 else float(t_tran)/t_succ))
    print()
    print("Simple acc: {}, \n"
          "sub_s acc: {}, \n"
          "Success on sub_s: {}, \n"
          "Transferred ex: {}, \n"
          "Transfer rate: {}".format(simple_a,
              subsimple_a, 
              s_succ, 
              s_tran,
              0 if s_succ==0 else float(s_tran)/s_succ))
    

if __name__ == "__main__":
    save_location = "models"
    os.system("mkdir -p {}".format(save_location))
    main()

