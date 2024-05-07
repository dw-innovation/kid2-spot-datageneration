echo Generate Tag List With Properties
python -m datageneration.retrieve_combinations \
--source datageneration/data/Primary_Keys_filtered10.xlsx \
--output_file datageneration/data/tag_combinations_v12.jsonl \
--generate_tag_list_with_properties

echo Generate Property Examples
python -m datageneration.retrieve_combinations \
--source datageneration/data/Primary_Keys_filtered10.xlsx \
--output_file datageneration/data/prop_examples_v12.jsonl \
--prop_example_limit 100000 \
--generate_property_examples
