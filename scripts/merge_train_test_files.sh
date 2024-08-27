python -m datageneration.merge_train_test_files \
--input_files datageneration/results/v15/train_fixed.tsv,datageneration/results/v15/dev_fixed.tsv,datageneration/results/v16/dev_non_roman_areas.tsv,datageneration/results/v16/dev_contain_rels.tsv,datageneration/results/v16/dev_case_props.tsv,datageneration/results/v16/train_non_roman_areas.tsv,datageneration/results/v16/train_contain_rels.tsv,datageneration/results/v16/train_case_props.tsv \
--output_folder datageneration/results/v16
