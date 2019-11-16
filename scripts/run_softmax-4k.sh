#!/bin/sh

source activate img-classifcation

EXP_NAME=softmax-4k

flags="--model softmax \
       --training-mode supervised \
       --not-full-data \
       --train-data-size 4000
       --optimiser sgd \
       --learning-rate 0.01 \
       --momentum 0.0 \
       --weight-decay 0.0 \
       --epochs 40 \
       --batch-size 512 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --log-interval 2 \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
