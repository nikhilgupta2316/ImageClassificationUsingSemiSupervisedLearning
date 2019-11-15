#!/bin/sh

source activate img-classifcation

flags="--model onelayernn \
       --data-aug \
       --optimiser adam \
       --learning-rate 0.001 \
       --lr-reducer \
       --lr-lambda-scheduler \
       --epochs 80 \
       --batch-size 64 \
       --train-data-size 49000\
       --exp-name resnet \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/onelayernn.log
