nohup python3.6 train.py --seed 1 --bS 10 --accumulate_gradients 2 \
	--bert_type_abb uS --fine_tune --lr 0.001 \
	--lr_bert 0.00001 --max_seq_leng 222  >log_no_constraint.log 2>err_no_constraint.log &
