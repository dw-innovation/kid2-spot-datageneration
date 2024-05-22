VERSION=v12

python -m datageneration.gpt_data_generator \
--relative_spatial_terms_path datageneration/data/relative_spatial_terms.csv \
--tag_query_file datageneration/results/${VERSION}/samples.jsonl \
--output_prompt_generations datageneration/results/${VERSION}/prompt_generations.jsonl \
--output_gpt_generations datageneration/results/${VERSION}/gpt_generations.jsonl \
--persona_path datageneration/prompts/personas.txt \
--styles_path datageneration/prompts/styles.txt \
--prob_usage_of_relative_spatial_terms 0.4 \
--prob_usage_of_written_numbers 0.3 \
--prob_of_typos 0.3 \
--max_dist_digits 5 \
--save_yaml_csv \
--generate_prompts
# Options: --generate_prompts , --generate_sentences