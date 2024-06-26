mkdir -p datageneration/results/v13/train_test
python -m datageneration.construct_train_test \
--input_folder datageneration/results/v13/gpt_generations \
--output_folder datageneration/results/v13/train_test \
--dev_samples 1000