#!/bin/sh

source activate img-classifcation

EXP_NAME=softmax-49k

flags="--model softmax \
       --train-data-size 49000 \
       --batch-size 512 \
       --epochs 160 \
       --data-aug \
       --optimiser sgd \
       --learning-rate 0.01 \
       --momentum 0.9 \
       --lr-reducer \
       --weight-decay 5e-4 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
