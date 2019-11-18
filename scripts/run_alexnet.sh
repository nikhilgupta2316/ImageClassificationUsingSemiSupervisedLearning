#!/bin/sh

source activate img-classifcation

EXP_NAME=alexnet-49k

flags="--model alexnet \
       --train-data-size 49000 \
       --batch-size 128 \
       --epochs 350 \
       --data-aug \
       --optimiser sgd \
       --learning-rate 0.001 \
       --lr-reducer \
       --weight-decay 5e-4 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
