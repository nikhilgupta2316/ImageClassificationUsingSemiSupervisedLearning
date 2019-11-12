#!/bin/sh

source activate img-classifcation

flags="--model softmax \
       --optimiser sgd \
       --learning-rate 0.01 \
       --momentum 0.0 \
       --weight-decay 0.0 \
       --epochs 5 \
       --batch-size 512 \
       --exp-name softmax \
       --tensorboard "

unbuffer python train.py $flags | tee checkpoints/softmax.log
