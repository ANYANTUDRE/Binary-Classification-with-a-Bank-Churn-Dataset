#!/bin/sh
python train.py --fold 0 --model lgbm
python train.py --fold 1 --model lgbm
python train.py --fold 2 --model lgbm
python train.py --fold 3 --model lgbm
python train.py --fold 4 --model lgbm
