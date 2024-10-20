#!/bin/bash

bank_train="bank-1/train.csv"
bank_test="bank-1/test.csv"

echo "Running AdaBoost"


python3 adaboost.py "$bank_train" "$bank_test" "500" >> adaboost_output.txt

echo "Running bagging"

python3 bagging.py "$bank_train" "$bank_test" "500" >> bagging_output.txt

echo "Running Random Forest"
python3 randomForest.py "$bank_train" "$bank_test" "500" >> ran_forest_output.txt
