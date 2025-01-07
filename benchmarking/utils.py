import json
import os
import pandas as pd
from datageneration.utils import split_descriptors


def write_output(generated_combs, output_file):
    """
    Writes the generated_combs to JSON with the given output_file path.

    :param generated_combs: The generated combinations.
    :param output_file: The path where the output file should be written.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb.model_dump(mode="json"), out_file)
            out_file.write('\n')

def load_key_table(path):
    """
    Loads the primary key table and transforms it into a map where each individual descriptor maps to a list of all
    descriptors in its bundle.

    :param path: The path to the primary key table file.
    :return: descriptors - Map of descriptors.
    """
    primary_key_table = pd.read_excel(path, engine='openpyxl')

    descriptors = {}
    for row in primary_key_table.to_dict(orient='records'):
        descriptors_str = row['descriptors']

        descriptors_lst = list(split_descriptors(descriptors_str))

        for desc in descriptors_lst:
            descriptors[desc] = descriptors_lst

    return descriptors