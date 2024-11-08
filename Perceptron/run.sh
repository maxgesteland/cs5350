#!/bin/bash

train_data="bank-note/train.csv"
test_data="bank-note/test.csv"

echo "Running Perceptron"
python3 perceptron.py "$train_data" "$test_data" "10" >> perceptron_output.txt

echo "Running Voted Perceptron"
python3 "perceptron-vote.py" "$train_data" "$test_data" "10" >> voted_perceptron_output.txt


echo "Running Average Perceptron"
python3 avg_perceptron.py "$train_data" "$test_data" "10" >> average_perceptron_output.txt
