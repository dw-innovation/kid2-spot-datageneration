import pandas as pd
import json

"""
Color Hex Validation Script

This script:
- Loads a dictionary of hex color codes and their natural language definitions.
- Loads a JSONL dataset of property examples (e.g., building or roof colors).
- Extracts all color values associated with specific color-related tags.
- Checks how many of these color values are missing from the color dictionary.

Useful for validating coverage of color descriptions in the dataset.
"""

if __name__ == '__main__':
    # Load color hex dictionary: maps hex codes to natural language color names
    with open('datageneration/data/color_encycolorpedia.json', 'r') as file:
        all_color_hex = json.load(file)
        all_color_hex = {key.lower(): value for key, value in all_color_hex.items()}

    # Define which color-related tags we want to evaluate
    color_tags = ['roof:colour=***example***', 'building:colour=***example***']

    # Load dataset of property examples
    color_file = 'datageneration/data/prop_examples_v17.jsonl'
    prop_data = pd.read_json(color_file, lines=True)

    # Extract all color examples where the tag is in our list
    examples_series = prop_data[prop_data['key'].isin(color_tags)]['examples']
    colors = set(sum(examples_series.tolist(), []))  # flatten and deduplicate

    print(f'Number of colors: {len(colors)}')

    # Check how many hex colors are missing from the dictionary
    hex_to_color_mapping = {}
    none_values = 0
    for color in colors:
        if color.startswith('#'):
            print(f'Checking {color}')
            lowerized_color = color.lower()
            nl_definition = all_color_hex.get(lowerized_color, None)
            print(nl_definition)
            if not nl_definition:
                none_values += 1
                print(lowerized_color)

    print(none_values)