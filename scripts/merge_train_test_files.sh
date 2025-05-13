#python -m datageneration.merge_train_test_files \
#--input_files datageneration/results/v15/train_fixed.tsv,datageneration/results/v15/dev_fixed.tsv,datageneration/results/v16_1/dev_props.tsv,datageneration/results/v16_1/train_props.tsv,datageneration/results/v16_2/train_props_v2.tsv,datageneration/results/v16_2/dev_props_v2.tsv \
#--output_folder datageneration/results/v16_2

#python -m datageneration.merge_train_test_files \
#--input_files datageneration/results/v15/train_fixed.tsv,datageneration/results/v15/dev_fixed.tsv,datageneration/results/v16_3/dev_props_v2.tsv,datageneration/results/v16_3/train_props_v2.tsv \
#--output_folder datageneration/results/v16_3

#python -m datageneration.merge_train_test_files \
#--input_files datageneration/results/v17/train_v17.tsv,datageneration/results/v17/dev_v17.tsv,datageneration/results/v17_3/train_v17_3.tsv,datageneration/results/v17_3/dev_v17_3.tsv \
#--output_folder datageneration/results/v17_3

#python -m datageneration.merge_train_test_files \
#--input_files datageneration/results/v17/train_v17-1-2.tsv,datageneration/results/v17/dev_v17-1-2.tsv,datageneration/results/v17_3/train_v17_3.tsv,datageneration/results/v17_3/dev_v17_3.tsv \
#--output_folder datageneration/results/v17_3


python -m datageneration.merge_train_test_files \
--input_files datageneration/results/v18_fullDataset-25k/train_test/train_v18.tsv,datageneration/results/v18_fullDataset-25k/train_test/dev_v18.tsv,datageneration/results/v18_fullDataset_part2_4k/train_test/train.tsv,datageneration/results/v18_fullDataset_part2_4k/train_test/dev.tsv,datageneration/results/v18_75k-filtered/train_test/train.tsv,datageneration/results/v18_75k-filtered/train_test/dev.tsv \
--output_folder datageneration/results/v18_2


