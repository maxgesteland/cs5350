To Run:

-put "bank-note" folder within Neural Networks project on the same level as run.sh and other python scripts
-run the run.sh script to generate the outputs in the following order
    -First it will ron the nn-backprop.py which is just the back and forward funcitonality and prints out the 
    weights from the written problem in the hw, its weights are hardcoded.

    -next the nn-sgd.py file which has the same backprop and forward functionality
    as the nn-backprop.py but it also does the stochiastic gradient descent funcitonality. the run.sh
    will output a nn-sgd.txt file with the associated training and test errors for all the widths

    -next the nn-sgd-0.py file which has the same functionality as the nn-sgd.py but has weights hardcoded to 0
    the run.sh will output a nn-sgd-0.txt file with the associated training and test errors for all the widths

    -Lastly the nn-bonus.py is my pytorch implementation that outputs the requested info for the hw
    into the nn-bonus.txt file.

running that script will output all the required info for the questions in the homework 