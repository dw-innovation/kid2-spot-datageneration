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
from benchmarking.utils import load_key_table
from benchmarking.entity_analyzer import EntityAndPropertyAnalyzer
from benchmarking.relation_analyzer import RelationAnalyzer
from benchmarking.area_analyzer import AreaAnalyzer
from typing import Dict

# class ResultDataType(enum.Enum):
#     TRUE = 'TRUE'
#     FALSE = 'FALSE'
#     NOT_APPLICABLE = 'NOT_APPLICABLE'
#     # PARTIAL_TRUE = 'PARTIALLY_TRUE'


class Result(BaseModel, frozen=True):
    yaml_true_string: str = Field(...)
    yaml_pred_string: str = Field(...)
    is_parsable_yaml: bool = Field(description="True if yaml can be parsed, otherwise False",
                                             default=False)
    is_perfect_match: bool = Field(description="True if area, entities+props and relations are equal, otherwise False",
                                                default=False)

    def __getitem__(self, item):
        return getattr(self, item)


def is_parsable_yaml(yaml_string) -> bool:
    """
    Checks whether the input batch of YAML strings is parsable.

    :return: is_parsable, parsed_yaml - Boolean whether YAML is parsable plus parsed YAML (or None if not possible).
    """
    is_parsable = False
    parsed_yaml = None
    try:
        parsed_yaml = yaml.safe_load(yaml_string)
        is_parsable = True
    except Exception as e:
        try:
            parsed_yaml = validate_and_fix_yaml(yaml_string)
        except Exception as e:
            pass
    return is_parsable, parsed_yaml


# def prepare_relation(data) -> ResultDataType:
#     """
#     In order to compare relations independent of the order of entities, it is not sufficient to have numeric
#     references for target and source. This method therefore replaces the numeric pointers with the descriptors (names)
#     of the references entities, as this makes comparisons possible.
#
#     :param data: The entire query, including area, entities and relations.
#     :return: prepped_relation - The updated relation with descriptors instead of numeric pointers.
#     """
#     relations = copy.deepcopy(data["relations"])
#     prepped_relation = copy.deepcopy(data["relations"])
#     for id in range(len(data["relations"])):
#         srcs = [ent["name"].lower() for ent in data["entities"] if ent["id"] == relations[id]["source"]]
#         if len(srcs) > 0:
#             prepped_relation[id]["source"] = srcs[0]
#         else:
#             prepped_relation[id]["source"] = "-1"
#         trgts = [ent["name"].lower() for ent in data["entities"] if ent["id"] == relations[id]["target"]]
#         if len(trgts) > 0:
#             prepped_relation[id]["target"] = trgts[0]
#         else:
#             prepped_relation[id]["target"] = "-1"
#
#     return prepped_relation


# def compare_relations(reference_relations, predicted_relations) -> ResultDataType:
#     """
#     Check if two lists of relations are identical. There are two different ways how the comparison is done, based on
#     whether the order of source and target is relevant or not (only the case in "contains" relations).
#     Contains relations (where the order matters) are compared as lists. Other relations (where the order of source
#     and target does not matter) is compared as a list of frozensets.
#
#     :param reference_relations: The first relations list to compare (ref_rel).
#     :param predicted_relations: The second relations list to compare (gen_rel).
#     :return: Boolean whether the two relations lists are the same.
#     """
#     if not predicted_relations:
#         return 0
#     total_relations = max(len(reference_relations), len(predicted_relations))
#     matches = 0
#     r1 = set()
#     r2 = set()
#     c1 = list()
#     c2 = list()
#     for id in range(len(reference_relations)):
#         if reference_relations[id]["type"] == "contains":
#             c1.append([reference_relations[id]["source"], reference_relations[id]["target"]])
#         elif reference_relations[id]["type"] == "dist" or reference_relations[id]["type"] == "distance":
#             # print('reference_relations!')
#             # print(reference_relations[id])
#             r1.add(frozenset({reference_relations[id]["source"], reference_relations[id]["target"], reference_relations[id]["value"]}))
#     for id in range(len(predicted_relations)):
#         # print(relations2[id])
#         if predicted_relations[id]["type"] == "contains":
#             c2.append([predicted_relations[id]["source"], predicted_relations[id]["target"]])
#         elif predicted_relations[id]["type"] == "dist" or predicted_relations[id]["type"] == "distance":
#             # print('predicted_relations!')
#             # print(predicted_relations[id])
#             r2.add(frozenset({predicted_relations[id]["source"], predicted_relations[id]["target"], predicted_relations[id]["value"]}))
#
#     r1 = Counter(r1)
#     r2 = Counter(r2)
#     for item in r1:
#         if item in r2:
#             matches += 1
#     c2_copy = copy.deepcopy(c2)
#     for c1_ in c1:
#         for id2, c2_ in enumerate(c2_copy):
#             if c1_ == c2_:
#                 c2_copy.pop(id2)
#                 matches += 1
#
#     return matches / total_relations

def normalize_name_brands(data):
    for entity in data.get('entities', []):
        if 'properties' in entity:
            for prop in entity['properties']:
                if 'brand' in prop.get('name', ''):
                    prop['name'] = 'name'
    return data


def compare_yaml(area_analyzer: AreaAnalyzer, entity_and_prop_analyzer: EntityAndPropertyAnalyzer, relation_analyzer: RelationAnalyzer, yaml_true_string, yaml_pred_string) -> Dict:
    """
    Compare two YAML structures represented as strings. This is done by comparing areas, entities and relations
    separately.

    :param yaml_true_string: The first YAML to compare.
    :param yaml_pred_string: The first YAML to compare.
    :return: Boolean whether the two YAMLs are the same.
    """
    _, ref_data = is_parsable_yaml(yaml_true_string)
    yaml_pred_string = yaml_pred_string.replace('</s>', '')
    _is_parsable_yaml, generated_data = is_parsable_yaml(yaml_pred_string)

    is_perfect_match = False
    ref_area = ref_data['area']
    gen_area = generated_data.get('area', None)

    results_area = area_analyzer.compare_area(ref_area, gen_area)
    ref_data = normalize_name_brands(ref_data)
    if generated_data:
        generated_data = normalize_name_brands(generated_data)

    ref_entities = ref_data.get('entities', None)
    gen_entities = None
    if generated_data:
        gen_entities = generated_data.get('entities', None)

    results_ents_props, full_paired_entities = entity_and_prop_analyzer.compare_entities(ref_entities,gen_entities)
    results_relations = relation_analyzer.compare_relations(ref_data, generated_data, full_paired_entities)
    if results_area['area_perfect_result'] and results_ents_props['props_perfect_result'] and results_ents_props['entity_perfect_result'] and results_relations['relation_perfect_result']:
        is_perfect_match = True

    # todo refactor this
    remaining_results = dict(yaml_pred_string=yaml_pred_string,
                  yaml_true_string=yaml_true_string,
                  is_perfect_match=is_perfect_match,
                  is_parsable_yaml=_is_parsable_yaml,
                             )
    all_results = remaining_results | results_area | results_ents_props | results_relations
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

    # todo: add colors
    meta_fields = ["1 entity", "2 entities", "3 entities", "distance relation", "area", "proporties",
                   "typos", "grammar mistakes", "rel spatial term", "cluster", "contains relation",
                   "brand/name as property", "brand/name standalone", "non-roman alphabet"]
    meta_results = dict.fromkeys(meta_fields, 0)
    meta_results_counter = dict.fromkeys(meta_fields, 0)

    predictions = pd.read_json(path_or_buf=pred_file_path, lines=True).to_dict(orient='records')

    gold_file_path = args.gold_file_path
    gold_sheet_name = args.gold_sheet_name
    gold_ds = pd.read_excel(gold_file_path, sheet_name=gold_sheet_name)
    gold_labels = gold_ds.to_dict(orient='records')

    area_analyzer = AreaAnalyzer()
    descriptors = load_key_table(path=args.key_table_path)
    entity_and_prop_analyzer = EntityAndPropertyAnalyzer(descriptors=descriptors)
    relation_analyzer = RelationAnalyzer()

    results = []
    for prediction, gold_label in tqdm(zip(predictions, gold_labels), total=len(gold_labels)):
        prediction['sentence'] = prediction['sentence'].strip().lower()
        gold_label['sentence'] = gold_label['sentence'].strip().lower()
        assert prediction['sentence'] == gold_label['sentence']

        yaml_pred_string = prediction['model_result']
        yaml_true_string = gold_label['YAML']

        result= {'sentence': prediction['sentence']}
        comparison_result = compare_yaml(
            area_analyzer=area_analyzer,
            relation_analyzer=relation_analyzer,
            entity_and_prop_analyzer=entity_and_prop_analyzer,
            yaml_true_string=yaml_true_string,
            yaml_pred_string=yaml_pred_string
        )

        result = result | comparison_result
        meta_vals = {key: gold_label[key] for key in meta_fields}
        results.append(result | meta_vals)

        for meta_field in meta_fields:
            if gold_label[meta_field] == 1:
                if result["is_perfect_match"]:
                    meta_results[meta_field] += 1
                    meta_results_counter[meta_field] += 1
                else:
                    meta_results_counter[meta_field] += 1

    for meta_field in meta_fields:
        if meta_results_counter[meta_field] == 0:
            del meta_results[meta_field]
        else:
            meta_results[meta_field] = meta_results[meta_field] / meta_results_counter[meta_field]

    evaluation_scores = {}
    results = pd.DataFrame(results)
    evaluation_scores['is_parsable_yaml'] = len(results[results['is_parsable_yaml'] == True]) / len(results)

    # Area Results
    total_area = results['total_area'].sum()
    total_bbox = results['total_bbox'].sum()
    total_name_area = results['total_name_area'].sum()
    evaluation_scores['percentage_correct_area'] = (results['num_correct_name_area'].sum()+ results['num_correct_bbox'].sum()) / total_area
    evaluation_scores['percentage_correct_bbox_area'] = results['num_correct_bbox'].sum() / total_bbox
    evaluation_scores['percentage_correct_name_area'] = results['num_correct_name_area'].sum() / total_name_area
    evaluation_scores['percentage_correct_area_type'] = results['num_correct_area_type'].sum() / total_area

    # Entity - Property Results
    total_ref_entities = results['total_ref_entities'].sum()
    evaluation_scores['entity_match_perfect_acc'] = results['num_entity_match_perfect'].sum() / total_ref_entities
    evaluation_scores['entity_match_weak_acc'] = results['num_entity_match_weak'].sum() / total_ref_entities
    evaluation_scores['percentage_correct_entity_type_acc'] = results['num_correct_entity_type'].sum() / total_ref_entities

    total_clusters = results['total_clusters'].sum()
    evaluation_scores['percentage_cluster_distance_acc'] = results['num_correct_cluster_distance'].sum() / total_clusters
    evaluation_scores['percentage_cluster_points_acc'] = results['num_correct_cluster_points'].sum() / total_clusters

    evaluation_scores["num_missing_entity"] = results['num_missing_entity'].sum()
    evaluation_scores["num_hallucinated_entity"] = results['num_hallucinated_entity'].sum()

    total_properties = results['total_properties'].sum()
    evaluation_scores['percentage_correct_properties_perfect_acc'] = results['num_correct_properties_perfect'].sum() / total_properties
    # evaluation_scores['percentage_correct_properties_weak_acc'] = results['num_correct_properties_weak'].sum() / total_properties

    total_height_property = results['total_height_property'].sum()
    evaluation_scores['percentage_correct_height_metric'] = results['num_correct_height_metric'].sum() / total_height_property
    evaluation_scores['percentage_correct_height_distance'] = results['num_correct_height_distance'].sum() / total_height_property
    evaluation_scores['percentage_correct_height'] = results['num_correct_height'].sum() / total_height_property

    total_cuisine_property = results['total_cuisine_property'].sum()
    evaluation_scores['percentage_correct_cuisine_property'] = results['num_correct_cuisine_properties'].sum() / total_cuisine_property

    total_color_property = results['total_color_property'].sum()
    evaluation_scores['percentage_correct_color_property'] = results['num_correct_color'].sum() / total_color_property

    evaluation_scores['num_hallucinated_properties'] = results['num_hallucinated_properties'].sum()
    evaluation_scores['num_missing_properties'] = results['num_missing_properties'].sum()

    # Relation Results
    total_dist_rels = results['total_dist_rels'].sum()
    total_contains_rels = results['total_contains_rels'].sum()
    total_relative_spatial_terms = results['total_relative_spatial_terms'].sum()
    total_rels = results['total_rels'].sum()

    evaluation_scores['percentage_correct_rel_type'] = results['num_correct_rel_type'].sum() / total_rels
    evaluation_scores['percentage_correct_dist_rels'] = results['num_correct_dist_rels'].sum() / total_dist_rels
    evaluation_scores['percentage_correct_contains_rels'] = results['num_correct_contains_rels'].sum() / total_contains_rels
    evaluation_scores['percentage_correct_dist_value'] = results['num_correct_dist_metric'].sum() / total_dist_rels
    evaluation_scores['percentage_correct_dist_metric'] = results['num_correct_dist_value'].sum() / total_dist_rels
    evaluation_scores['percentage_correct_dist'] = results['num_correct_dist'].sum() / total_dist_rels
    evaluation_scores['percentage_correct_relative_spatial_terms'] = results['num_correct_relative_spatial_terms'].sum() / total_relative_spatial_terms

    evaluation_scores = evaluation_scores | meta_results

    for eval_type, eval_value in evaluation_scores.items():
        print(f'==={eval_type}===')
        print(eval_value)
    evaluation_scores = evaluation_scores | meta_results
    evaluation_scores = pd.DataFrame(evaluation_scores, index=[0])
    with pd.ExcelWriter(out_file_path) as writer:
        results.to_excel(writer)
    with pd.ExcelWriter(out_file_path_sum) as writer:
        evaluation_scores.to_excel(writer)
