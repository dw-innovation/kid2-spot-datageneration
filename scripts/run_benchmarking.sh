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

#echo llama3 v15 with case Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/results/llama3_v1_18082024.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v15_1_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama3_training_ds_v15_1_8082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 v15 Results

#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v15.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v15_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama3_training_ds_v15_8082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 v16.1 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16_1.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16_1_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama3_training_ds_v16_1_8082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 v16.2 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16_2.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16_2_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama3_training_ds_v16_2_8082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 v16.3 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_18082024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16_3.jsonl
#GOLD_SHEET_NAME=gold_annotations_18082024
#OUT_FILE_PATH=benchmarking/results/spot_llama3_training_ds_v16_3_18082024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama3_training_ds_v16_3_8082024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 spot_Meta-Llama-3.1-8B_ep10_training_ds_v17-1-2 Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_llama-3-8b_ep10_training_ds_v16_3-17_1-2.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/spot_llama-3-8b_ep10_training_ds_v16_3-17_1-2_05112024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama-3-8b_ep10_training_ds_v16_3-17_1-2_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM


#echo llama3 spot_llama-3-8b_ep10_training_ds_v17-1-2_param-6.jsonl Results
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_llama-3-8b_ep10_training_ds_v17-1-2_param-6.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/spot_llama-3-8b_ep10_training_ds_v17-1-2_param-6_05112024_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_llama-3-8b_ep10_training_ds_v17-1-2_param-6-2_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 spot_llama-3-8b_ep10_training_ds_v17-1-2_param-5 Results
#
#FNAME=spot_llama-3-8b_ep10_training_ds_v17-1-2_param-5
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo llama3 spot_llama-3-8b_ep10_training_ds_v17-1-2_param-4 Results
#FNAME=spot_llama-3-8b_ep10_training_ds_v17-1-2_param-4
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo spot_Meta-Llama-3.1-8B_ep10_training_ds_v16_3-17_1-2 Results
#FNAME=spot_Meta-Llama-3.1-8B_ep10_training_ds_v16_3-17_1-2
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo spot_llama-3-8b_ep10_training_ds_v17-1-2_3_param-6 Results
#FNAME=spot_llama-3-8b_ep10_training_ds_v17-1-2_3_param-6
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#echo spot_llama-3-8b_ep10_training_ds_v17-1-2_3_param-6 Results
#FNAME=spot_llama-3-8b_ep10_training_ds_v17-1-2_3_param-6
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#FNAME=gold_predictions_gpt-4o_cot_zeroshot_2
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#FNAME=spot_Mistral-Nemo-Base-2407_ep10_training_ds_v16_3-17_1-2_param-6
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

FNAME=spot_llama-3-8b_ep10_training_ds_v16_3-17_1-2_lora
KEY_TABLE_PATH=datageneration/data/Spot_primary_keys_bundles.xlsx
GOLD_FILE_PATH=benchmarking/data/gold_annotations_12052025.xlsx
PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
GOLD_SHEET_NAME=gold_annotations_12052025
OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_12052025_eval-summary.xlsx

python -m benchmarking.evaluate_results  \
--key_table_path $KEY_TABLE_PATH \
--gold_file_path $GOLD_FILE_PATH \
--pred_file_path $PRED_FILE_PATH \
--gold_sheet_name $GOLD_SHEET_NAME \
--out_file_path $OUT_FILE_PATH \
--out_file_path_sum $OUT_FILE_PATH_SUM

#FNAME=gold_predictions_gpt-4o_cot_fewshot_1_2
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_05112024_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM

#MODEL='llama-3-8b' # llama-3-8b # Meta-Llama-3.1-8B
#VERSION=v16_3-17_1-2  #v16_3-17_1-2
#DATE=15112024
#EPOCHS=10
#PARAMETER_VERSION=5
#echo Llama3 ds_${VERSION} Results, ${DATE}
#
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_05112024.xlsx
#PRED_FILE_PATH=benchmarking/results/spot_${MODEL}_ep${EPOCHS}_training_ds_${VERSION}_param-${PARAMETER_VERSION}.jsonl
#echo $PRED_FILE_PATH
#GOLD_SHEET_NAME=gold_annotations_05112024
#OUT_FILE_PATH=benchmarking/results/spot_${MODEL}_${EPOCHS}_training_ds_${VERSION}_param-${PARAMETER_VERSION}_${DATE}_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/spot_${MODEL}_${EPOCHS}_training_ds_${VERSION}_param-${PARAMETER_VERSION}_${DATE}_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM