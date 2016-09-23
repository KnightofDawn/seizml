#!/bin/bash

CLFS='rf l1log l2log ada svc'
echo $CLFS

foo () {
    local clf=$1
    exe1="train_test_s1_"$clf".py"
    exe2="train_test_s2_"$clf".py"
    echo $exe1
    echo $exe2
    python ./$exe1
    python ./$exe2
#    echo "train_cv_s1_"$clf".py"
}

for clf in $CLFS; do foo "$clf" & done

