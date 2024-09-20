#!/bin/bash

car_train="car/train.csv"
car_test="car/test.csv"

split_methods=("entropy" "gini" "majority_error")

max_depths=(1 2 3 4 5 6)

for method in "${split_methods[@]}"; do
    for depth in "${max_depths[@]}"; do
        echo "$method depth: $depth"
        python3 decisionTree.py "$car_train" "$car_test" "$method" "$depth"
        echo ""
    done
done

bank_train="bank/train.csv"
bank_test="bank/test.csv"

max_depths_bank=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

echo "bank"
for method in "${split_methods[@]}"; do
    for depth in "${max_depths_bank[@]}"; do
        echo "$method depth: $depth"
        python3 decisionTree.py "$bank_train" "$bank_test" "$method" "$depth"
        echo ""
    done
done
