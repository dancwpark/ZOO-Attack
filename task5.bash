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

first_line="noise,samps,query_count,ASR,distortion"
attack="black"
num_imgs=11
untargeted=""
prefix="T"
# for dataset in mnist cifar10
# do
#     FILENAME="./results/task5/${prefix}_$dataset.out"
#     echo $first_line > $FILENAME
#     for noise in 0.01 0.1 0.25
#     do
#         for samps in 10 50 100
#         do
#             single_line=$(python test_all.py -d $dataset -a $attack -N $noise -D -k $samps -b 9 -n $num_imgs --solver adam $untargeted | grep L5)
#             A="$(cut -d',' -f3 <<< $single_line)"
#             C="$(cut -d',' -f5 <<< $single_line)"
#             D="$(cut -d',' -f7 <<< $single_line)"
#             data=$noise
#             data+=","
#             data+=$samps
#             data+=","
#             data+=$A
#             data+=","
#             data+=$C
#             data+=","
#             data+=$D

#             echo $data
#             echo $data >> $FILENAME
#         done
#     done
# done


dataset="imagenet"
FILENAME="./results/task5/${prefix}_$dataset.out"
echo $first_line > $FILENAME
for noise in 0.01 0.1 0.25
do
    for samps in 10 50 100
    do 
        single_line=$(python test_all.py -d $dataset -a $attack -N $noise -D -k $samps -f 123 --solver adam -b 1 -c 10.0 --use_resize --reset_adam -m 3000 -n $num_imgs $untargeted | grep L5)
        A="$(cut -d',' -f3 <<< $single_line)"
        C="$(cut -d',' -f5 <<< $single_line)"
        D="$(cut -d',' -f7 <<< $single_line)"
        data=$noise
        data+=","
        data+=$samps
        data+=","
        data+=$A
        data+=","
        data+=$C
        data+=","
        data+=$D

        echo $data
        echo $data >> $FILENAME
   done
done