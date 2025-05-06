mkdir -p datageneration/results/v18_fullDataset_part2/train_test
python -m datageneration.construct_train_test \
--input_file datageneration/results/v18_fullDataset_part2/gpt_generations_dataset_v18_fullDataset_part2_25k_yaml.jsonl \
--output_folder datageneration/results/v18_fullDataset_part2/train_test \
--dev_samples 1000