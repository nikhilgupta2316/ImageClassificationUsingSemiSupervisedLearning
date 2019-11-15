#!/bin/sh

source activate img-classifcation

flags="--model alexnet \
       --data-aug \
       --optimiser sgd \
       --learning-rate 0.001 \
       --epochs 100 \
       --batch-size 128 \
       --exp-name alexnet \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/alexnet.log