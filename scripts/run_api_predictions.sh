FNAME=gpt_oss_remote
GOLD_SHEET_NAME=descriptor_updates_13052026
BENCHMARKING_FILE=benchmarking/data/goldstandard_testing_dataset.xlsx
OUTPUT_FILE=benchmarking/results/${FNAME}_14072026.jsonl
ENVIRONMENT=development

python -m benchmarking.run_api_predictions \
--output_file $OUTPUT_FILE \
--benchmarking_file $BENCHMARKING_FILE \
--gold_sheet_name $GOLD_SHEET_NAME \
--environment $ENVIRONMENT
