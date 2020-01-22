#!/bin/bash

# arguments for test_all.py
# -d : select dataset ["mnist", "cifar10", "imagenet"]
# -a : select attack ["white","black"]
# -n : number of images to test [1-100]
# -b : # of binary search steps (this should always be 9)
# -u : set this flag for untargeted attack, if you leave it out it will run a targeted attack over all other classes
# -D : flag to activate the defense
# -N : set the noise level of the defense [>0]
#
# if in doubt about what arguments to set, look at the example attacks here: 
# https://github.com/IBM/ZOO-Attack 
# (Especially for imagenet, as it requires special settings to be effective). or ask me :)

first_line="query_count,ASR,distortion"
attack="black"
num_imgs=11
untargeted=""
prefix="T"
for dataset in mnist cifar10
do
    FILENAME="./results/task6/${prefix}_$dataset.out"
    echo $first_line > $FILENAME
    # Run against distillation defense
    # mnist and cifar10
    # black box attack
    # do not use logits -- but we are not applying our defense
    single_line=$(python test_all-distilled.py -d $dataset -a $attack -z -b 9 -n $num_imgs --solver adam $untargeted | grep L5)
    A="$(cut -d',' -f3 <<< $single_line)"
    C="$(cut -d',' -f5 <<< $single_line)"
    D="$(cut -d',' -f7 <<< $single_line)"
    data=$A
    data+=","
    data+=$C
    data+=","
    data+=$D

    echo $data
    echo $data >> $FILENAME
done
