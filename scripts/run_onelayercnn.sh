#!/bin/sh

source activate img-classifcation

EXP_NAME=onelayercnn

flags="--model onelayercnn \
       --data-aug \
       --optimiser adam \
       --learning-rate 0.001 \
       --lr-reducer \
       --epochs 80 \
       --batch-size 64 \
       --train-data-size 49000\
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
