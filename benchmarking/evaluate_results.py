import copy
import enum
import numpy as np
import pandas as pd
import yaml
from argparse import ArgumentParser
from pydantic import BaseModel, Field
from tqdm import tqdm
from collections import Counter

from benchmarking.utils import write_output
from benchmarking.yaml_parser import validate_and_fix_yaml
from benchmarking.utils import load_key_table, check_equivalent_entities
from benchmarking.entity_analyzer import EntityAndPropertyAnalyzer
from typing import Dict

class ResultDataType(enum.Enum):
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    # PARTIAL_TRUE = 'PARTIALLY_TRUE'


class Result(BaseModel, frozen=True):
    yaml_true_string: str = Field(...)
    yaml_pred_string: str = Field(...)
    is_parsable_yaml: ResultDataType = Field(description="True if yaml can be parsed, otherwise False",
                                             default=ResultDataType.FALSE)

    is_perfect_match: ResultDataType = Field(description="True if area, entities+props and relations are equal, otherwise False",
                                                default=ResultDataType.FALSE)

    is_area_match: ResultDataType = Field(description="True if areas are equal, otherwise False",
                                                default=ResultDataType.FALSE)

    are_entities_exactly_same: ResultDataType = Field(description="True if entity are equal, otherwise False",
                                                      default=ResultDataType.FALSE)

    percentage_entities_exactly_same: float = Field(
        description="Percentage of corectly identified entities over the total ents",
        default=0.0)

    are_entities_same_exclude_props: ResultDataType = Field(description="True if entity are equal, otherwise False",
                                                            default=ResultDataType.FALSE)

    percentage_entities_same_exclude_props: float = Field(
        description="Percentage of corectly identified entities over the total ents, exclude props",
        default=0.0)

    are_relations_exactly_same: ResultDataType = Field(description="True if relations are equal, otherwise False",
                                                       default=ResultDataType.NOT_APPLICABLE)

    percentage_relations_same: float = Field(
        description="Percentage of corectly identified entities over the total ents, exclude props",
        default=0.0)

    are_properties_same: ResultDataType = Field(description="True if relations are equal, otherwise False",
                                                       default=ResultDataType.NOT_APPLICABLE)

    percentage_properties_same: float = Field(
        description="Percentage of corectly identified entities over the total ents",
        default=0.0)

    percentage_correct_entity_type: float = Field(descrtiption = 'Average number of the entity type match', default=0.0)
    # are_entities_partially_same: ResultDataType = Field(description = 'Partially some entities match with the reference data')

    def __getitem__(self, item):
        return getattr(self, item)


class AreaAnalyzer:
    def __init__(self):
        pass

    def compare_areas_strict(self, ref_area, test_area) -> ResultDataType:
        """
        Checks if two areas are identical.

        :param area1: The first area to compare.
        :param area2: The second area to compare.
        :return: Boolean whether the two areas are the same.
        """
        return ResultDataType.TRUE if (ref_area == test_area) else ResultDataType.FALSE

    def compare_areas_light(self, ref_area, test_area) -> ResultDataType:
        """
        Checks if two areas are identical.

        :param area1: The first area to compare.
        :param area2: The second area to compare.
        :return: Boolean whether the two areas are the same.
        """
        if ref_area["type"] != "bbox":
            if test_area['type'] == "bbox":
                return ResultDataType.FALSE
            ref_area['value'] = ref_area['value'].lower()
            if 'value' in test_area:
                test_area['value'] = test_area['value'].lower()
            else:
                test_area['value'] = test_area['name'].lower()

        else:
            # generations sometimes omit the value
            print(ref_area)
            print(test_area)
            if ref_area['type'] == test_area['type']:
                return ResultDataType.TRUE

        # todo: relaxing encoding issue

        return self.compare_areas_strict(ref_area=ref_area, test_area=test_area)

def is_parsable_yaml(yaml_string) -> ResultDataType:
    """
    Checks whether the input batch of YAML strings is parsable.

    :return: is_parsable, parsed_yaml - Boolean whether YAML is parsable plus parsed YAML (or None if not possible).
    """
    parsed_yaml = None
    try:
        parsed_yaml = yaml.safe_load(yaml_string)
        is_parsable = ResultDataType.TRUE
    except Exception as e:
        is_parsable = ResultDataType.FALSE
        # try to parse it by using the custom parser from the backend
        try:
            parsed_yaml = validate_and_fix_yaml(yaml_string)
        except Exception as e:
            pass
    return is_parsable, parsed_yaml


def prepare_relation(data) -> ResultDataType:
    """
    In order to compare relations independent of the order of entities, it is not sufficient to have numeric
    references for target and source. This method therefore replaces the numeric pointers with the descriptors (names)
    of the references entities, as this makes comparisons possible.

    :param data: The entire query, including area, entities and relations.
    :return: prepped_relation - The updated relation with descriptors instead of numeric pointers.
    """
    relations = copy.deepcopy(data["relations"])
    prepped_relation = copy.deepcopy(data["relations"])
    for id in range(len(data["relations"])):
        srcs = [ent["name"].lower() for ent in data["entities"] if ent["id"] == relations[id]["source"]]
        if len(srcs) > 0:
            prepped_relation[id]["source"] = srcs[0]
        else:
            prepped_relation[id]["source"] = "-1"
        trgts = [ent["name"].lower() for ent in data["entities"] if ent["id"] == relations[id]["target"]]
        if len(trgts) > 0:
            prepped_relation[id]["target"] = trgts[0]
        else:
            prepped_relation[id]["target"] = "-1"

    return prepped_relation


def compare_relations(reference_relations, predicted_relations) -> ResultDataType:
    """
    Check if two lists of relations are identical. There are two different ways how the comparison is done, based on
    whether the order of source and target is relevant or not (only the case in "contains" relations).
    Contains relations (where the order matters) are compared as lists. Other relations (where the order of source
    and target does not matter) is compared as a list of frozensets.

    :param reference_relations: The first relations list to compare (ref_rel).
    :param predicted_relations: The second relations list to compare (gen_rel).
    :return: Boolean whether the two relations lists are the same.
    """
    if not predicted_relations:
        return 0
    total_relations = max(len(reference_relations), len(predicted_relations))
    matches = 0
    r1 = set()
    r2 = set()
    c1 = list()
    c2 = list()
    for id in range(len(reference_relations)):
        if reference_relations[id]["type"] == "contains":
            c1.append([reference_relations[id]["source"], reference_relations[id]["target"]])
        elif reference_relations[id]["type"] == "dist" or reference_relations[id]["type"] == "distance":
            # print('reference_relations!')
            # print(reference_relations[id])
            r1.add(frozenset({reference_relations[id]["source"], reference_relations[id]["target"], reference_relations[id]["value"]}))
    for id in range(len(predicted_relations)):
        # print(relations2[id])
        if predicted_relations[id]["type"] == "contains":
            c2.append([predicted_relations[id]["source"], predicted_relations[id]["target"]])
        elif predicted_relations[id]["type"] == "dist" or predicted_relations[id]["type"] == "distance":
            # print('predicted_relations!')
            # print(predicted_relations[id])
            r2.add(frozenset({predicted_relations[id]["source"], predicted_relations[id]["target"], predicted_relations[id]["value"]}))

    r1 = Counter(r1)
    r2 = Counter(r2)
    for item in r1:
        if item in r2:
            matches += 1
    c2_copy = copy.deepcopy(c2)
    for c1_ in c1:
        for id2, c2_ in enumerate(c2_copy):
            if c1_ == c2_:
                c2_copy.pop(id2)
                matches += 1

    return matches / total_relations

def normalize_name_brands(data):
    for entity in data.get('entities', []):
        if 'properties' in entity:
            for prop in entity['properties']:
                if 'brand' in prop.get('name', ''):
                    prop['name'] = 'name'
    return data


def compare_yaml(key_table_path: str, area_analyzer: AreaAnalyzer, entity_and_prop_analyzer: EntityAndPropertyAnalyzer, yaml_true_string, yaml_pred_string) -> Dict:
    """
    Compare two YAML structures represented as strings. This is done by comparing areas, entities and relations
    separately.

    :param yaml_true_string: The first YAML to compare.
    :param yaml_pred_string: The first YAML to compare.
    :return: Boolean whether the two YAMLs are the same.
    """
    _, ref_data = is_parsable_yaml(yaml_true_string)
    _is_parsable_yaml, generated_data = is_parsable_yaml(yaml_pred_string)
    is_perfect_match = ResultDataType.FALSE
    is_area_match = ResultDataType.FALSE
    # are_entities_exactly_same = ResultDataType.FALSE
    # percentage_entities_exactly_same = -1.0
    # are_entities_same_exclude_props = ResultDataType.FALSE
    # percentage_entities_same_exclude_props = -1.0
    are_relations_exactly_same = ResultDataType.NOT_APPLICABLE
    percentage_relations_same = -1.0
    # are_properties_same = ResultDataType.NOT_APPLICABLE
    # percentage_properties_same = -1.0
    # num_entities_on_ref_data: int = 0
    # num_entities_on_gen_data: int = 0
    num_relations_on_ref_data: int = 0
    num_relations_on_gen_data: int = 0
    are_entities_partially_same = ResultDataType.FALSE

    if generated_data:
        # num_entities_on_ref_data = len(ref_data['entities'])
        # num_entities_on_gen_data = len(generated_data['entities'])
        if "relations" in ref_data:
            num_relations_on_ref_data = len(ref_data['relations'])
        else:
            num_relations_on_ref_data = -1.0
        if "relations" in generated_data:
            num_relations_on_gen_data = len(generated_data['relations'])
        else:
            num_relations_on_gen_data = -1.0

        is_area_match = area_analyzer.compare_areas_light(ref_data['area'], generated_data['area'])
        ref_data = normalize_name_brands(ref_data)
        generated_data = normalize_name_brands(generated_data)

        # descriptors = load_key_table(key_table_path)
        # generated_data['entities'] = check_equivalent_entities(descriptors, ref_data['entities'],
        #                                                        generated_data['entities'])

        results_ents_props = entity_and_prop_analyzer.compare_entities(ref_data['entities'],
                                                                            generated_data['entities'])

        # if percentage_entities_exactly_same == 1.0:
        #     are_entities_exactly_same = ResultDataType.TRUE
        # elif percentage_properties_same == 0.0:
        #     are_entities_exactly_same = ResultDataType.FALSE
        # # elif percentage_entities_exactly_same!=0.0:
        # #     are_entities_partially_same = ResultDataType.TRUE
        #
        # if percentage_entities_same_exclude_props == 1.0:
        #     are_entities_same_exclude_props = ResultDataType.TRUE
        #
        # ref_entities_with_properties = {}
        # for ref_entity in ref_data['entities']:
        #     if 'properties' in ref_entity:
        #         ref_entities_with_properties[ref_entity['name']] = ref_entity['properties']
        #
        # if len(ref_entities_with_properties) > 0:
        #     predicted_entities_with_properties = {}
        #     for pred_entity in generated_data['entities']:
        #         if 'properties' in pred_entity:
        #             predicted_entities_with_properties[pred_entity['name']] = pred_entity['properties']
        #
        #     percentage_properties_same = property_analyzer.percentage_properties_same(
        #         ref_entities=ref_entities_with_properties, prop_entities=predicted_entities_with_properties)

        # if percentage_properties_same == 1.0:
        #     are_properties_same = ResultDataType.TRUE
        # elif percentage_properties_same == -1.0:
        #     are_properties_same = ResultDataType.NOT_APPLICABLE
        # else:
        #     are_properties_same = ResultDataType.FALSE


        # todo: recheck this!!
        if 'relations' not in ref_data:
            are_relations_exactly_same = ResultDataType.NOT_APPLICABLE

        else:
            if 'relations' not in generated_data:
                are_relations_exactly_same = ResultDataType.FALSE

            else:
                ref_relations_prepared = prepare_relation(ref_data)
                generated_relations_prepared = prepare_relation(generated_data)
                percentage_relations_same = compare_relations(ref_relations_prepared, generated_relations_prepared)
                if percentage_relations_same == 1.0:
                    are_relations_exactly_same = ResultDataType.TRUE
                else:
                    are_relations_exactly_same = ResultDataType.FALSE

    if (is_area_match == ResultDataType.TRUE and results_ents_props['num_entity_match_perfect']
        and (are_relations_exactly_same == ResultDataType.TRUE or
             are_relations_exactly_same == ResultDataType.NOT_APPLICABLE)):
        is_perfect_match = ResultDataType.TRUE



    # todo refactor this
    remaining_results = dict(yaml_pred_string=yaml_pred_string,
                  yaml_true_string=yaml_true_string,
                  is_perfect_match=is_perfect_match,
                  is_parsable_yaml=_is_parsable_yaml,
                  is_area_match=is_area_match,
                  num_relations_on_ref_data=num_relations_on_ref_data,
                  num_relations_on_gen_data=num_relations_on_gen_data,
                  are_relations_exactly_same=are_relations_exactly_same,
                  percentage_relations_same=percentage_relations_same)

    all_results = remaining_results | results_ents_props
    return all_results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--key_table_path', type=str, required=True)
    parser.add_argument('--gold_file_path', type=str, required=True)
    parser.add_argument('--gold_sheet_name', type=str, required=True)
    parser.add_argument('--pred_file_path', type=str, required=True)
    parser.add_argument('--out_file_path', type=str, required=True)
    parser.add_argument('--out_file_path_sum', type=str, required=True)
    # parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    args = parser.parse_args()
    # geolocations_file_path = args.geolocations_file_path
    key_table_path = args.key_table_path
    out_file_path = args.out_file_path
    out_file_path_sum = args.out_file_path_sum
    pred_file_path = args.pred_file_path

    meta_fields = ["1 entity", "2 entities", "3 entities", "distance relation", "area", "proporties",
                   "typos", "grammar mistakes", "rel spatial term", "cluster", "contains relation",
                   "brand/name as property", "brand/name standalone", "non-roman alphabet"]
    meta_results = dict.fromkeys(meta_fields, 0)
    meta_results_counter = dict.fromkeys(meta_fields, 0)

    predictions = pd.read_json(path_or_buf=pred_file_path, lines=True).to_dict(orient='records')

    gold_file_path = args.gold_file_path
    gold_sheet_name = args.gold_sheet_name
    gold_labels = pd.read_excel(gold_file_path, sheet_name=gold_sheet_name).to_dict(orient='records')

    area_analyzer = AreaAnalyzer()
    entity_and_prop_analyzer = EntityAndPropertyAnalyzer()

    results = []
    for prediction, gold_label in tqdm(zip(predictions, gold_labels), total=len(gold_labels)):
        prediction['sentence'] = prediction['sentence'].strip().lower()
        gold_label['sentence'] = gold_label['sentence'].strip().lower()

        assert prediction['sentence'] == gold_label['sentence']


        yaml_pred_string = prediction['model_result']

        # print(yaml_pred_string)


        yaml_true_string = gold_label['YAML']
        result= {'sentence': prediction['sentence']}
        comparision_result = compare_yaml(key_table_path=key_table_path,
                              area_analyzer=area_analyzer,
                              entity_and_prop_analyzer=entity_and_prop_analyzer,
                              yaml_true_string=yaml_true_string,
                              yaml_pred_string=yaml_pred_string)
        result = result | comparision_result
        meta_vals = {key: gold_label[key] for key in meta_fields}
        results.append(result | meta_vals)

        for meta_field in meta_fields:
            if gold_label[meta_field] == 1:
                if result["is_perfect_match"] == ResultDataType.TRUE:
                    meta_results[meta_field] += 1
                    meta_results_counter[meta_field] += 1
                else:
                    meta_results_counter[meta_field] += 1

    for meta_field in meta_fields:
        if meta_results_counter[meta_field] == 0:
            del meta_results[meta_field]
        else:
            meta_results[meta_field] = meta_results[meta_field] / meta_results_counter[meta_field]

    results = pd.DataFrame(results)

    print(results.columns)

    evaluation_scores = {}

    # Results with binary type
    for result_type in ['is_perfect_match',
                        'is_parsable_yaml',
                        'is_area_match',
                        'are_entities_exactly_same',
                        # 'are_entities_partially_same',
                        'percentage_entities_exactly_same',
                        'percentage_correct_entity_type',
                        'are_entities_same_exclude_props',
                        'percentage_entities_same_exclude_props',
                        'are_properties_same',
                        'percentage_properties_same',
                        'are_relations_exactly_same',
                        'percentage_relations_same']:
        print(f"===Results for {result_type}===")

        if result_type in ['percentage_entities_exactly_same',
                        'percentage_entities_same_exclude_props',
                        'percentage_correct_entity_type',
                        'percentage_properties_same',
                        'percentage_relations_same']:
            na_samples = results[results[result_type] == -1]
            valid_results = results[results[result_type] != -1]
            acc = np.mean(valid_results[result_type].to_numpy())
        else:
            na_samples = results[results[result_type] == ResultDataType.NOT_APPLICABLE]
            true_preds = results[results[result_type] == ResultDataType.TRUE]
            acc = len(true_preds) / (len(results) - len(na_samples))

        evaluation_scores[result_type + "_acc"] = acc
        print(f'  Accuracy of {result_type}: {acc}')

        if result_type in ["are_relations_exactly_same", "are_properties_same"]:
            evaluation_scores[result_type + "_NA"] = len(na_samples)
            print(f"  Number of NA samples: {len(na_samples)}")

    evaluation_scores = evaluation_scores | meta_results

    evaluation_scores = pd.DataFrame(evaluation_scores, index=[0])

    def convert_custom_type(value):
        if value == ResultDataType.TRUE:
            return "True"
        elif value == ResultDataType.FALSE:
            return "False"
        elif value == ResultDataType.NOT_APPLICABLE:
            return "Not Applicable"
        return value

    results = results.map(convert_custom_type)

    print(evaluation_scores)

    with pd.ExcelWriter(out_file_path) as writer:
        results.to_excel(writer)
    with pd.ExcelWriter(out_file_path_sum) as writer:
        evaluation_scores.to_excel(writer)
