#!/bin/sh

source activate img-classifcation

EXP_NAME=softmax

flags="--model softmax \
       --optimiser sgd \
       --learning-rate 0.01 \
       --momentum 0.0 \
       --weight-decay 0.0 \
       --epochs 20 \
       --batch-size 512 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
