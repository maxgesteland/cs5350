#!/bin/bash

train_data="concrete/train.csv"
test_data="concrete/test.csv"

echo "Running batch descent"

python3 batchDescent.py "$train_data" "$test_data" >> batch_output.txt

echo "Running sgd"
python3 sgd.py "$train_data" "$test_data" >> sgd_output.txt

echo "Running weight vector calculation in analytical way"
python3 analytical.py "$train_data" >> analytical_output.txt