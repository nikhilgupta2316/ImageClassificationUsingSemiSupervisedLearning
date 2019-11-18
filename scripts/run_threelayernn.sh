#!/bin/sh

source activate img-classifcation

EXP_NAME=threelayernn-2

flags="--model threelayernn \
       --train-data-size 49000 \
       --batch-size 64 \
       --epochs 240 \
       --data-aug \
       --optimiser adam \
       --learning-rate 0.001 \
       --lr-reducer \
       --weight-decay 5e-6 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
