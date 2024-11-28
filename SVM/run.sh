#!/bin/bash

train_data="bank-note/train.csv"
test_data="bank-note/test.csv"
T=100

echo "Running SVM with schedule 1 (gamma_t = gamma_0 / (1 + gamma_0 / a * t))" >> primal-output.txt

C_values=(100/873 500/873 700/873)
gamma=(0.1)
a=(1)

for C_expr in "${C_values[@]}"
do
    C=$(echo "scale=5; $C_expr" | bc)

    echo "C=$C_expr, gamma_0=$gamma, a=$a" >> primal-output.txt
    python3 svm-primal.py "$train_data" "$test_data" "$T" "$C" "$gamma" "$a" "1" >> primal-output.txt
done

echo "Running SVM with schedule 2 (gamma_t = gamma_0 / (1 + t))" >> primal-output.txt

for C_expr in "${C_values[@]}"
do
    C=$(echo "scale=5; $C_expr" | bc)
    echo "C=$C_expr, gamma_0=$gamma" >> primal-output.txt
    python3 svm-primal.py "$train_data" "$test_data" "$T" "$C" "0.1" "1" "2" >> primal-output.txt
done



# Now, run SVM Dual
echo "Running SVM Dual" >> dual-output.txt

for C_expr in "${C_values[@]}"
do
    C=$(echo "scale=5; $C_expr" | bc)
    echo "C=$C_expr" >> dual-output.txt
    python3 svm-dual.py "$train_data" "$test_data" "$C" >> dual-output.txt
done


echo "Running Kernel SVM with Gaussian Kernel" > kernel-output.txt

gamma_values=(0.1 0.5 1 5 100)

for gamma in "${gamma_values[@]}"
do
    for C_expr in "${C_values[@]}"
    do
        C=$(echo "scale=5; $C_expr" | bc)
        echo -n "C=$C_expr, gamma=$gamma " >> kernel-output.txt
        python3 svm-kernel.py "$train_data" "$test_data" "$C" "$gamma" >> kernel-output.txt
    done
done