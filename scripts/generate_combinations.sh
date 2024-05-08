VERSION=v12

python -m datageneration.generate_combination_table \
--geolocations_file_path datageneration/data/countries+states+cities.json \
--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
--output_file datageneration/results/${VERSION}/samples.jsonl \
--write_output \
--max_distance_digits 5 \
--max_number_of_entities_in_prompt 4 \
--max_number_of_props_in_entity 4 \
--percentage_of_entities_with_props 0.3 \
--percentage_of_two_word_areas 0.5 \
--samples 500