#!/bin/sh

source activate img-classifcation

flags="--model onelayercnn \
       --data-aug \
       --optimiser adam \
       --learning-rate 0.001 \
       --lr-reducer \
       --epochs 80 \
       --batch-size 64 \
       --train-data-size 49000\
       --exp-name onelayercnn \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/onelayercnn.log