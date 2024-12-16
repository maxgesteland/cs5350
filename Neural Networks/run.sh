#!/bin/bash

train_data="bank-note/train.csv"
test_data="bank-note/test.csv"


input_vector="1 1 1 1"

h1=3
h2=3

python3 nn-backprop.py "$input_vector" $h1 $h2 > nn-backprop.txt

widths=(5 10 25 50 100)

gamma_0=0.1
d=25
epochs=10

for h in "${widths[@]}"; do
    echo "width: $h" >> nn-sgd.txt
    python3 nn-sgd.py "$train_data" "$test_data" $h $gamma_0 $d $epochs >> nn-sgd.txt
done


widths=(5 10 25 50 100)

gamma_0=0.1
d=25
epochs=10

for h in "${widths[@]}"; do
    echo "width: $h" >> nn-sgd-0.txt
    python3 nn-sgd-0.py "$train_data" "$test_data" $h $gamma_0 $d $epochs >> nn-sgd-0.txt
done

activations=("tanh" "relu")
depths=(3 5 9)
widths=(5 10 25 50 100)
epochs=10  

for act in "${activations[@]}"; do
    echo "activation: $act" >>nn-bonus.txt
    for dpt in "${depths[@]}"; do
        for w in "${widths[@]}"; do
            echo "depth: $dpt, width: $w" >> nn-bonus.txt
            python3 nn-bonus.py "$train_data" "$test_data" "$act" $dpt $w $epochs >> nn-bonus.txt
        done
    done
done