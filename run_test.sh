echo "dev"
python3.6 evaluate_ws.py --source_file data/wikisql_tok/dev.jsonl \
                         --db_file data/wikisql_tok/dev.db \
                         --pred_file saved/results_dev.jsonl
echo "test"
python3.6 evaluate_ws.py --source_file data/wikisql_tok/test.jsonl \
                         --db_file data/wikisql_tok/test.db \
                         --pred_file saved/results_test.jsonl
