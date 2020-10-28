#!/bin/sh -l
# FILENAME:  run.sh
module load anaconda
source activate py36
cd /home/maosen/sqlova
source ~/.bashrc
python3 train.py --seed 1 --bS 16 --accumulate_gradients 2 --bert_type_abb uL --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222 
