#!/bin/sh

source activate img-classifcation

EXP_NAME=alexnet

flags="--model alexnet \
       --data-aug \
       --optimiser sgd \
       --learning-rate 0.001 \
       --epochs 100 \
       --batch-size 128 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
