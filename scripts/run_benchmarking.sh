FNAME=spot_gpt-oss-20b-unsloth_ep5_training_ds_v18_120fix_75k_param-7_prompt-v2
KEY_TABLE_PATH=datageneration/data/Spot_primary_keys_bundles.xlsx
GOLD_FILE_PATH=benchmarking/data/gold_annotations_14072025.xlsx
PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
GOLD_SHEET_NAME=gold_annotations_14072025
OUT_FILE_PATH=benchmarking/results/${FNAME}_25082025_eval.xlsx
OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_25082025_eval-summary.xlsx

python -m benchmarking.evaluate_results  \
--key_table_path $KEY_TABLE_PATH \
--gold_file_path $GOLD_FILE_PATH \
--pred_file_path $PRED_FILE_PATH \
--gold_sheet_name $GOLD_SHEET_NAME \
--out_file_path $OUT_FILE_PATH \
--out_file_path_sum $OUT_FILE_PATH_SUM