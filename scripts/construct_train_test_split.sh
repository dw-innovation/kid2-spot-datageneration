mkdir -p datageneration/results/v18_120fix_75k/train_test
python -m datageneration.construct_train_test \
--input_file datageneration/results/v18_120fix_75k/gpt_generations_dataset_v18_120fix_75k_yaml.jsonl \
--output_folder datageneration/results/v18_120fix_75k/train_test \
--dev_samples 3500