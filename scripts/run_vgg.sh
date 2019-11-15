#!/bin/sh

source activate img-classifcation

flags="--model vggnet \
       --optimiser sgd \
       --learning-rate 0.01 \
       --momentum 0.9 \
       --weight-decay 5e-4 \
       --epochs 350 \
       --batch-size 128 \
       --test-batch-size 128 \
       --data-aug \
       --random-crop-size 32 \
       --random-crop-pad 4 \
       --exp-name vgg \
       --tensorboard "

unbuffer python train.py $flags | tee checkpoints/vgg.log