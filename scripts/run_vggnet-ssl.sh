#!/bin/sh

source activate img-classifcation

EXP_NAME=vggnet-ssl

flags="--model vggnet \
       --training-mode semi-supervised \
       --train-data-size 4000 \
       --batch-size 128 \
       --ssl-label-generation-batch-size 64 \
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
