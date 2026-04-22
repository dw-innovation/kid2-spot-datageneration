#mkdir -p datageneration/results/v18_120fix_75k/train_test
#python -m datageneration.construct_train_test \
#--input_file datageneration/results/v18_120fix_75k/gpt_generations_dataset_v18_120fix_75k_yaml.jsonl \
#--output_folder datageneration/results/v18_120fix_75k/train_test \
#--dev_samples 3500

# mkdir -p datageneration/results/v18_120fix_75k/train_test
# python -m datageneration.construct_train_test \
# --input_file datageneration/results/v19_newBundles_75k/gpt_generations_dataset_v19_newBundles_75k_temp_yaml.jsonl \
# --output_folder datageneration/results/v19_newBundles_75k/ \
# --dev_samples 3500

# python -m datageneration.construct_train_test \
# --input_file datageneration/results/v19_newBundles_10k/gpt_generations_dataset_v19_newBundles_10k_yaml.jsonl \
# --output_folder datageneration/results/v19_newBundles_10k/ \
# --dev_samples 1000

python -m datageneration.construct_train_test \
--input_file datageneration/results/v19_newBundles_10k_1ent/gpt_generations_dataset_v19_newBundles_10k_1ent_yaml.jsonl \
--output_folder datageneration/results/v19_newBundles_10k_1ent/ \
--dev_samples 1000
