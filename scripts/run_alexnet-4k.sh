#!/bin/sh

source activate img-classifcation

EXP_NAME=alexnet-4k

flags="--model alexnet \
       --training-mode supervised \
       --not-full-data \
       --train-data-size 4000 \
       --batch-size 128 \
       --epochs 350 \
       --data-aug \
       --optimiser sgd \
       --learning-rate 0.001 \
       --lr-reducer \
       --weight-decay 5e-4 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --log-interval 2 \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
