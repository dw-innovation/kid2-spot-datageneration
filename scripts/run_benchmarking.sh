#FNAME=spot_gpt-oss-20b-unsloth_ep5_training_ds_v18_120fix_75k_param-7_prompt-v2
#KEY_TABLE_PATH=datageneration/data/Spot_primary_keys_bundles.xlsx
#GOLD_FILE_PATH=benchmarking/data/gold_annotations_14072025.xlsx
#PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
#GOLD_SHEET_NAME=gold_annotations_14072025
#OUT_FILE_PATH=benchmarking/results/${FNAME}_25082025_eval.xlsx
#OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_25082025_eval-summary.xlsx
#
#python -m benchmarking.evaluate_results  \
#--key_table_path $KEY_TABLE_PATH \
#--gold_file_path $GOLD_FILE_PATH \
#--pred_file_path $PRED_FILE_PATH \
#--gold_sheet_name $GOLD_SHEET_NAME \
#--out_file_path $OUT_FILE_PATH \
#--out_file_path_sum $OUT_FILE_PATH_SUM



# FNAME=spot_mistralai_Mistral-Small-3.2-24B-Instruct-2506_ep5_training_ds_v18_120fix_75_v19_25k_35k_param-7_prompt-v2
# KEY_TABLE_PATH=SPOT_OSM-tag-bundles-UPDATED.xlsx
# GOLD_FILE_PATH=benchmarking/data/goldstandard_testing_dataset.xlsx
# PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
# GOLD_SHEET_NAME=descriptor_updates_02022026
# OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
# OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_eval-summary.xlsx

# python -m benchmarking.evaluate_results  \
# --key_table_path $KEY_TABLE_PATH \
# --gold_file_path $GOLD_FILE_PATH \
# --pred_file_path $PRED_FILE_PATH \
# --gold_sheet_name $GOLD_SHEET_NAME \
# --out_file_path $OUT_FILE_PATH \
# --out_file_path_sum $OUT_FILE_PATH_SUM

# FNAME=gpt_oss_remote
# KEY_TABLE_PATH=/home/barisschlichti/Dokumente/Codes/kid2-spot-datageneration/benchmarking/data/SPOT_OSM-tag-bundles-UPDATED-workInProgress.xlsx
# GOLD_FILE_PATH=benchmarking/data/goldstandard_testing_dataset.xlsx
# PRED_FILE_PATH=benchmarking/results/${FNAME}.jsonl
# GOLD_SHEET_NAME=descriptor_updates_13052026
# OUT_FILE_PATH=benchmarking/results/${FNAME}_eval.xlsx
# OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_eval-summary.xlsx

# python -m benchmarking.evaluate_results  \
# --key_table_path $KEY_TABLE_PATH \
# --gold_file_path $GOLD_FILE_PATH \
# --pred_file_path $PRED_FILE_PATH \
# --gold_sheet_name $GOLD_SHEET_NAME \
# --out_file_path $OUT_FILE_PATH \
# --out_file_path_sum $OUT_FILE_PATH_SUM


FNAME=gpt_oss_remote_14072026
KEY_TABLE_PATH=/home/barisschlichti/Dokumente/Codes/kid2-spot-datageneration/benchmarking/data/SPOT_OSM-tag-bundles-UPDATED-workInProgress.xlsx
GOLD_FILE_PATH=benchmarking/data/goldstandard_testing_dataset.xlsx
PRED_FILE_PATH=benchmarking/results/${FNAME}_2.jsonl
GOLD_SHEET_NAME=descriptor_updates_13052026
OUT_FILE_PATH=benchmarking/results/${FNAME}_eval_2.xlsx
OUT_FILE_PATH_SUM=benchmarking/results/${FNAME}_eval-summary_2.xlsx

python -m benchmarking.evaluate_results  \
--key_table_path $KEY_TABLE_PATH \
--gold_file_path $GOLD_FILE_PATH \
--pred_file_path $PRED_FILE_PATH \
--gold_sheet_name $GOLD_SHEET_NAME \
--out_file_path $OUT_FILE_PATH \
--out_file_path_sum $OUT_FILE_PATH_SUM
