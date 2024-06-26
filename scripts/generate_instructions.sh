VERSION=v13

python -m benchmarking.generate_instructions \
  --geolocations_file_path datageneration/data/countries+states+cities.json \
--tag_combination_path datageneration/data/tag_combinations_${VERSION}.jsonl \
--tag_prop_examples_path datageneration/data/prop_examples_${VERSION}.jsonl \
--relative_spatial_terms_path datageneration/data/relative_spatial_terms.csv
#--dev_samples 500