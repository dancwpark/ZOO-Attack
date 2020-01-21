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

first_line="noise,query_count,ASR,distortion"
attack="white"
num_imgs=11
untargeted=""
for dataset in imagenet
do
    FILENAME="./results/task2/T_$dataset.out"
    echo $first_line > $FILENAME
   
    for noise in 0 0.0001 0.001 0.01 0.1 0.25 0.5 1 10
    do
        single_line=$(python test_all.py -d $dataset -a $attack -N $noise -D -b 9 -n $num_imgs --solver adam $untargeted | grep L5)
        echo $single_line
        A="$(cut -d',' -f3 <<< $single_line)"
        C="$(cut -d',' -f5 <<< $single_line)"
        D="$(cut -d',' -f7 <<< $single_line)"
        data=$noise
        data+=","
        data+=$A
        data+=","
        data+=$C
        data+=","
        data+=$D
        echo $data >> $FILENAME
    done
done 
