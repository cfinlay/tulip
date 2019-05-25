#!/bin/sh

# Example script to train CIFAR-10 model with squared norm gradient regularization

MODEL='ResNeXt34_2x32'

# Setup
TIMESTAMP=`date +%y-%m-%dT%H%M%S`  # Use this in LOGDIR
DATADIR='/path/to/data/'           # Where datasets are stored

BASELOG='./logs/cifar10'/$MODEL
LOGDIR=$BASELOG/'L2-lambda-0.1-'$TIMESTAMP
SCRATCH='/path/to/scratch-drive/'$TIMESTAMP

mkdir -p $DATADIR
mkdir -p $SCRATCH
chmod g+rwx $SCRATCH
mkdir -p $BASELOG

ln -s $SCRATCH $LOGDIR


CUDA_VISIBLE_DEVICES=0 \
python -u ./train.py $DATADIR \
    --norm L2 \
    --penalty 0.1 \
    --model $MODEL \
    --logdir $LOGDIR \
    | tee $LOGDIR/log.out 2>&1

rm $LOGDIR
mv $SCRATCH $LOGDIR
