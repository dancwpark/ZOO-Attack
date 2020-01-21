#!/bin/bash

single_line=""
FILENAME=""
first_line="noise"

for dataset in mnist cifar10 imagenet
do
    FILENAME="./results/task1/$dataset.out"

    for column_names in {1..29} 
    do
      first_line+=",$column_names"
    done
    first_line+=",30"  
    echo $first_line > $FILENAME 
    
    for noise in 0.001 0.01 0.1 1 10 100
    do
        single_line=$noise
        for i in {1..30} 
        do  
            single_line+=","     
            single_line+=$(python verify.py -d $dataset -D -n $noise -b 100)
        done
        echo $single_line >> $FILENAME 
    done
done
