#echo mt5 Results

GOLD_FILE_PATH=benchmarking/results/gold_annotations_15072024.xlsx
PRED_FILE_PATH=benchmarking/results/mt5_tuned_base_minimized_v1_db-v13_output_yaml_out_gold_annotations_15072024.jsonl
OUT_FILE_PATH=benchmarking/results/mt5_tuned_base_minimized_v1_db-v13_output_yaml_out_gold_annotations_15072024_eval.xlsx

python -m benchmarking.evaluate_results \
--gold_file_path $GOLD_FILE_PATH \
--pred_file_path $PRED_FILE_PATH \
--gold_sheet_name gold_annotations_15072024 \
--out_file_path $OUT_FILE_PATH

#echo llama3 Results
#
#GOLD_FILE_PATH=benchmarking/results/gold_annotations_15072024.xlsx
#PRED_FILE_PATH=benchmarking/results/llama3_v1_01082024.jsonl
#OUT_FILE_PATH=benchmarking/results/llama3_v1_01082024_15072024_eval.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name gold_annotations_15072024 \
#--out_file_path $OUT_FILE_PATH