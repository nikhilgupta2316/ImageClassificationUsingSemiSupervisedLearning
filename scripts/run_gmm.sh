​
#!/bin/sh
​
source activate img-classification
​
EXP_NAME=gmm
flags="--model softmax \
       --exp-name ${EXP_NAME} \
       --training-mode gmm\
       --filelogger "

python train.py $flags