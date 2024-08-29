#echo mt5 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_08082024.xlsx
#PRED_FILE_PATH=benchmarking/data/mt5_tuned_base_minimized_v1_db-v13_output_yaml_out_08082024.jsonl
#GOLD_SHEET_NAME=gold_annotations_08082024
#OUT_FILE_PATH=benchmarking/results/mt5_tuned_base_minimized_v1_db-v13_output_yaml_out_13082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/mt5_tuned_base_minimized_v1_db-v13_output_yaml_out_13082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/data/llama3_v1_18082024.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/llama3_v1_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/llama3_v1_18082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/results/mt5_tuned_base_minimized_v1_db-v15_output_yaml_out_xx.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/mt5_tuned_base_minimized_v1_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/mt5_tuned_base_minimized_v1_18082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 v16 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama3_training_ds_v16_8082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

echo llama3 v15 Results

GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
PRED_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v15.jsonl
GOLD_SHEET_NAME=gold_annotations_18082024
OUT_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v15_18082024_eval.xlsx
OUT_FILE_PATH_SUM=benchmarking/results/spot_llama3_training_ds_v15_8082024_eval-summary.xlsx

python -m benchmarking.evaluate_results \
--gold_file_path $GOLD_FILE_PATH \
--pred_file_path $PRED_FILE_PATH \
--gold_sheet_name $GOLD_SHEET_NAME \
--out_file_path $OUT_FILE_PATH \
--out_file_path_sum $OUT_FILE_PATH_SUM
