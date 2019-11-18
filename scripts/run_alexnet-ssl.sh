#!/bin/sh

source activate img-classifcation

EXP_NAME=alexnet-ssl

flags="--model alexnet \
       --training-mode semi-supervised \
       --train-data-size 4000 \
       --batch-size 128 \
       --ssl-label-generation-batch-size 64 \
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
