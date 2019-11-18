#!/bin/sh

source activate img-classifcation

EXP_NAME=vggnet-49k

flags="--model vggnet \
       --train-data-size 49000 \
       --batch-size 128 \
       --test-batch-size 128 \
       --epochs 350 \
       --data-aug \
       --random-crop-size 32 \
       --random-crop-pad 4 \
       --optimiser sgd \
       --learning-rate 0.01 \
       --momentum 0.9 \
       --weight-decay 5e-4 \
       --exp-name ${EXP_NAME} \
       --tensorboard \
       --filelogger "

unbuffer python train.py $flags | tee checkpoints/${EXP_NAME}.log
