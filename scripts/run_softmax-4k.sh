#!/bin/sh

source activate img-classifcation

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
       --exp-name softmax-4k \
       --tensorboard \
       --log-interval 2 \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/softmax-4k.log
