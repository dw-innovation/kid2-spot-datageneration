import copy
import csv
import itertools
import json
import yaml
from pathlib import Path
from typing import List

SEPERATORS = ['=', '>', '~']

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

        yaml_string = yaml.dump(query)

        comb["query"] = yaml_string

    return new_combs


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
