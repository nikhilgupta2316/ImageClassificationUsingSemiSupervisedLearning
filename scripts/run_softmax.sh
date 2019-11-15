#!/bin/sh

source activate img-classifcation

flags="--model softmax \
       --optimiser sgd \
       --learning-rate 0.01 \
       --momentum 0.0 \
       --weight-decay 0.0 \
       --epochs 10 \
       --batch-size 512 \
       --exp-name softmax \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/softmax.log
