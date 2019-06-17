#!/bin/bash
DEVICE = 0
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 0

## conv
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 6

## linear++ + conv
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 7.121
## linear + conv
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 7.11

## linear++
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 7.12
## linear
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 7.1


#python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 5 # conv no one-hot

## film simple input
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 4

## one-hot like v2
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 3

## 45 + 1 concat
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 2

## 1 + 1 concat
python train.py -c configs/hpe/hpe_cond.yaml -d $DEVICE -ac 1