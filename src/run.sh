#!/bin/sh

python3 train.py --fold 0 --model decision_tree_gini
python3 train.py --fold 1 --model decision_tree_entropy
python3 train.py --fold 2 --model rf
python3 train.py --fold 3 --model rf
python3 train.py --fold 4 --model rf
