#VERSION=v18
#
#echo Generate Tag List With Properties
#python -m datageneration.retrieve_combinations \
#--source datageneration/data/Spot_primary_keys_bundles.xlsx \
#--output_file datageneration/data/tag_combinations_${VERSION}.jsonl \
#--prop_limit 100 \
#--min_together_count 5000 \
#--generate_tag_list_with_properties
#
#echo Generate Property Examples
#python -m datageneration.retrieve_combinations \
#--source datageneration/data/Spot_primary_keys_bundles.xlsx \
#--output_file datageneration/data/prop_examples_${VERSION}.jsonl \
#--prop_example_limit 50000 \
#--generate_property_examples \
#--add_non_roman_examples


VERSION=v19

echo Generate Tag List With Properties
python -m datageneration.retrieve_combinations \
--source SPOT_OSM-tag-bundles-UPDATED.xlsx \
--output_file datageneration/data/tag_combinations_${VERSION}.jsonl \
--prop_limit 100 \
--min_together_count 5000 \
--generate_tag_list_with_properties

#echo Generate Property Examples
#python -m datageneration.retrieve_combinations \
#--source SPOT_OSM-tag-bundles-UPDATED.xlsx \
#--output_file datageneration/data/prop_examples_${VERSION}.jsonl \
#--prop_example_limit 50000 \
#--generate_property_examples \
#--add_non_roman_examples
