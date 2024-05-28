import copy
import csv
import itertools
import json
import yaml
import numpy as np
from random import randint
from pathlib import Path
from typing import List

SEPERATORS = ['=', '>', '~']


# numerical value generator
def get_random_decimal_with_metric(max_digits: int) -> str:
    digits = randint(1, max_digits)
    low = np.power(10, digits - 1)
    high = np.power(10, digits) - 1
    num = randint(low, high)
    if np.random.choice([True, False], 1)[0]:
        num = num / np.random.choice([10, 100], 1)[0]

    dist = str(num) + " " + np.random.choice(["cm", "m", "km", "in", "ft", "yd", "mi"], 1)[0]

    return dist


def get_random_integer(max_digits: int) -> int:
    digits = randint(1, max_digits)
    low = np.power(10, digits - 1)
    high = np.power(10, digits) - 1

    return randint(low, high)


def add_yaml_to_filename(output_file):
    parent_dir = Path(output_file).parent
    filename_without_extension = Path(output_file).stem
    file_extension = Path(output_file).suffix
    yaml_output_file = parent_dir / (filename_without_extension + "_yaml" + file_extension)
    return yaml_output_file


def write_output(generated_combs, output_file):
    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb.model_dump(mode="json"), out_file)
            out_file.write('\n')


def write_dict_output(generated_combs, output_file, bool_add_yaml=True):
    if bool_add_yaml:
        output_file = add_yaml_to_filename(output_file)

    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb, out_file)
            out_file.write('\n')


def write_output_csv(generated_combs, output_file, bool_add_yaml=True):
    if bool_add_yaml:
        output_file = add_yaml_to_filename(output_file)

    parent_dir = Path(output_file).parent
    filename_without_extension = Path(output_file).stem
    new_output_file = parent_dir / (filename_without_extension + ".csv")

    keys = generated_combs[0].keys()
    with open(new_output_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(generated_combs)


def translate_queries_to_yaml(combs):
    new_combs = [c.dict() for c in combs]

    for comb in new_combs:
        query = comb["query"]

        query = clean_up_query(query)

        yaml_string = yaml.dump(query)

        comb["query"] = yaml_string

    return new_combs


<<<<<<< gpt_gen_with_yaml
def clean_up_query(query):
    for entity in query["entities"]:
        if len(entity["properties"]) == 0:
            entity.pop('properties', None)
        else:
            for property in entity["properties"]:
                if property["operator"] is None and property["value"] is None:
                    property.pop('operator', None)
                    property.pop('value', None)
    query["relations"] = query["relations"]["relations"]
    if query["relations"] is None:
        query.pop('relations', None)
    else:
        for relation in query["relations"]:
            if relation["value"] is None:
                relation.pop('value', None)
    return query
=======
def split_descriptors(descriptors: str) -> List[str]:
    '''this function splits the descriptors as a list of single descriptor'''
    processed_descriptors = set()

    for descriptor in descriptors.split('|'):
        descriptor = descriptor.lstrip().strip().lower()
        if len(descriptor) == 0:
            continue
        processed_descriptors.add(descriptor)

    return processed_descriptors

>>>>>>> main

class CompoundTagPropertyProcessor:
    def expand_list(self, tag_compounds: str) -> List[str]:
        processed_tag_compounds = []
        tag_compounds = tag_compounds.split('|')
        for tag_compound in tag_compounds:
            tag_compound = tag_compound.replace('[', '').replace(']', '').replace('"', '')
            if len(tag_compound) != 0:
                processed_tag_compounds.append(tag_compound)
        return processed_tag_compounds

    def run(self, tag_compounds: str) -> List[str]:
        selected_seperator = None

        for seperator in SEPERATORS:
            _tag_compounds = tag_compounds.split(seperator)

            if len(_tag_compounds) == 2:
                tag_compounds_keys = _tag_compounds[0]
                tag_compounds_values = _tag_compounds[1]
                selected_seperator = seperator
            else:
                continue

        assert selected_seperator

        if '[' in tag_compounds_keys:
            tag_compounds_keys = self.expand_list(tag_compounds_keys)

        if '[' in tag_compounds_values:
            tag_compounds_values = self.expand_list(tag_compounds_values)

        if isinstance(tag_compounds_values, str):
            tag_compounds_values = [tag_compounds_values]

        if isinstance(tag_compounds_keys, str):
            tag_compounds_keys = [tag_compounds_keys]

        processed_tag_compounds = []
        for tag_key, tag_value in itertools.product(tag_compounds_keys, tag_compounds_values):
            processed_tag_compounds.append(f'{tag_key.lower()}{selected_seperator}{tag_value.lower()}')

        return processed_tag_compounds
