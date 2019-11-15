#!/bin/sh

source activate img-classifcation

flags="--model resnet \
       --optimiser adam \
       --learning-rate 0.001 \
       --lr-reducer \
       --lr-lambda-scheduler \
       --epochs 40 \
       --batch-size 64 \
       --exp-name resnet \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/resnet.log