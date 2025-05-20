mkdir -p datageneration/results/v18_fullDataset_part2/train_test
python -m datageneration.construct_train_test \
--input_file datageneration/results/v18_41mini_50k/gpt_generations_dataset_v18_41mini_50k_25k_temp_yaml.jsonl \
--output_folder datageneration/results/v18_41mini_50k/train_test \
--dev_samples 3500