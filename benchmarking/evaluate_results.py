import copy
import enum
import numpy as np
import pandas as pd
import yaml
from argparse import ArgumentParser
from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarking.utils import write_output
from benchmarking.yaml_parser import validate_and_fix_yaml


class ResultDataType(enum.Enum):
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    NOT_APPLICABLE = 'NOT_APPLICABLE'


class Result(BaseModel, frozen=True):
    yaml_true_string: str = Field(...)
    yaml_pred_string: str = Field(...)
    is_parsable_yaml: ResultDataType = Field(description="True if yaml can be parsed, otherwise False",
                                             default=ResultDataType.FALSE)

    is_perfect_match: ResultDataType = Field(description="True if area, entities+props and relations are equal, otherwise False",
                                                default=ResultDataType.FALSE)


    is_area_exact_match: ResultDataType = Field(description="True if areas are equal, otherwise False",
                                                default=ResultDataType.FALSE)

    is_area_light_match: ResultDataType = Field(description="True if areas are equal, otherwise False",
                                                default=ResultDataType.FALSE)
    num_entities_on_ref_data: int = 0
    num_entities_on_gen_data: int = 0
    num_relations_on_ref_data: int = 0
    num_relations_on_gen_data: int = 0

    are_entities_exactly_same: ResultDataType = Field(description="True if entity are equal, otherwise False",
                                                      default=ResultDataType.FALSE)

    are_entities_same_exclude_props: ResultDataType = Field(description="True if entity are equal, otherwise False",
                                                            default=ResultDataType.FALSE)

    percentage_entities_partial_match_exclude_props: float = Field(
        description="Percentage of corectly identified entities over the total ents",
        default=0.0)

    are_relations_exactly_same: ResultDataType = Field(description="True if entity are equal, otherwise False",
                                                       default=ResultDataType.NOT_APPLICABLE)

    percentage_of_correctly_identified_properties: float = Field(
        description="Percentage of corectly identified entities over the total ents",
        default=0.0)

    def __getitem__(self, item):
        return getattr(self, item)


class AreaAnalyzer:
    def __init__(self):
        pass

    def compare_areas_strict(self, area1, area2) -> ResultDataType:
        """
        Checks if two areas are identical.

        :param area1: The first area to compare.
        :param area2: The second area to compare.
        :return: Boolean whether the two areas are the same.
        """
        return ResultDataType.TRUE if (area1 == area2) else ResultDataType.FALSE

    def compare_areas_light(self, area1, area2) -> ResultDataType:
        """
        Checks if two areas are identical.

        :param area1: The first area to compare.
        :param area2: The second area to compare.
        :return: Boolean whether the two areas are the same.
        """
        if area1["type"] != "bbox":
            area1['value'] = area1['value'].lower()
            area2['value'] = area2['value'].lower()

        else:
            # generations sometimes omit the value
            if area1['type'] == area2['type']:
                return ResultDataType.TRUE

        # todo: relaxing encoding issue

        return self.compare_areas_strict(area1=area1, area2=area2)


class PropertyAnalyzer:
    def __init__(self):
        pass

    def convert_values_to_string(self, data):
        for item in data:
            item["name"] = item["name"].lower()
            if 'value' not in item:
                continue
            if isinstance(item['value'], (int, float)):
                item['value'] = str(item['value'])
            else:
                item['value'] = item['value'].lower()

        return data

    def compare_properties(self, props1, props2) -> ResultDataType:
        """
        Check if two lists of properties are identical. The lists are first sorted via their names, to make sure the order
        does not affect the results.

        :param props1: The first property list to compare.
        :param props2: The second property list to compare.
        :return: Boolean whether the two property lists are the same.
        """
        if len(props1) != len(props2):
            return ResultDataType.FALSE
        props1 = self.convert_values_to_string(props1)
        props2 = self.convert_values_to_string(props2)
        props1_sorted = sorted(props1, key=lambda x: x['name'])
        props2_sorted = sorted(props2, key=lambda x: x['name'])

        return ResultDataType.TRUE if (props1_sorted == props2_sorted) else ResultDataType.FALSE

    def percentage_of_correctly_identified_properties(self, ref_entities, prop_entities):
        total_props = len(ref_entities)
        correctly_identified_properties = 0
        for ent, props in ref_entities.items():
            if ent not in prop_entities:
                continue

            result = self.compare_properties(props1=props, props2=prop_entities[ent])
            if result == ResultDataType.TRUE:
                correctly_identified_properties += 1

        return correctly_identified_properties / total_props


class EntityAnalyzer:
    def __init__(self, property_analyzer: PropertyAnalyzer):
        self.property_analyzer = property_analyzer

    def compare_entities_strict(self, entities1, entities2) -> ResultDataType:
        """
        Check if two lists of entities are identical. The lists are first sorted via their names, to make sure the order
        does not affect the results.

        :param entities1: The first entity list to compare.
        :param entities2: The second entity list to compare.
        :return: Boolean whether the two entity lists are the same.
        """
        if len(entities1) != len(entities2):
            return ResultDataType.FALSE
        entities1_sorted, entities2_sorted = self.sort_entities(entities1, entities2)
        for ent1, ent2 in zip(entities1_sorted, entities2_sorted):
            # if 'type' not in ent1:
            #     return ResultDataType.FALSE
            if ent1['name'].lower() != ent2['name'].lower() or ent1['type'] != ent2['type']:
                return ResultDataType.FALSE

            properties_check_results = property_analyzer.compare_properties(ent1.get('properties', []),
                                                                            ent2.get('properties', []))

            if properties_check_results == ResultDataType.FALSE:
                return ResultDataType.FALSE

        return ResultDataType.TRUE

    def sort_entities(self, entities1, entities2):
        entities1_sorted = sorted(entities1, key=lambda x: x['name'].lower())
        entities2_sorted = sorted(entities2, key=lambda x: x['name'].lower())
        return entities1_sorted, entities2_sorted

    def compare_entities_exclude_props(self, entities1, entities2) -> ResultDataType:
        if len(entities1) != len(entities2):
            return ResultDataType.FALSE
        entities1_sorted, entities2_sorted = self.sort_entities(entities1, entities2)
        for ent1, ent2 in zip(entities1_sorted, entities2_sorted):
            # if 'type' not in ent1:
            #     return ResultDataType.FALSE
            if ent1['name'].lower() != ent2['name'].lower() or ent1['type'] != ent2['type']:
                return ResultDataType.FALSE

        return ResultDataType.TRUE

    def compare_entities_partial_match_exclude_props(self, entities1, entities2) -> ResultDataType:
        """
        Check if two lists of entities are identical. The lists are first sorted via their names, to make sure the order
        does not affect the results.

        :param entities1: The first entity list to compare.
        :param entities2: The second entity list to compare.
        :return: Boolean whether the two entity lists are the same.
        """
        entities1_sorted, entities2_sorted = self.sort_entities(entities1, entities2)

        total_ents = len(entities1_sorted)
        matched_ents = 0
        for ent1, ent2 in zip(entities1_sorted, entities2_sorted):
            if ent1['name'].lower() == ent2['name'].lower() and ent1['type'] == ent2['type']:
                matched_ents += 1
        return matched_ents / total_ents


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
        prepped_relation[id]["source"] = \
            [ent["name"].lower() for ent in data["entities"] if ent["id"] == relations[id]["source"]][0]
        prepped_relation[id]["target"] = \
            [ent["name"].lower() for ent in data["entities"] if ent["id"] == relations[id]["target"]][0]
    return prepped_relation


def compare_relations(relations1, relations2) -> ResultDataType:
    """
    Check if two lists of relations are identical. There are two different ways how the comparison is done, based on
    whether the order of source and target is relevant or not (only the case in "contains" relations).
    Contains relations (where the order matters) are compared as lists. Other relations (where the order of source
    and target does not matter) is compared as a list of frozensets.

    :param relations1: The first relations list to compare.
    :param relations2: The second relations list to compare.
    :return: Boolean whether the two relations lists are the same.
    """
    if len(relations1) != len(relations2):
        return ResultDataType.FALSE
    r1 = set()
    r2 = set()
    c1 = list()
    c2 = list()
    for id in range(len(relations1)):
        if relations1[id]["type"] == "contains":
            c1.append([relations1[id]["source"], relations1[id]["target"]])
        elif relations1[id]["type"] == "dist" or relations1[id]["type"] == "distance":
            r1.add(frozenset({relations1[id]["source"], relations1[id]["target"], relations1[id]["value"]}))
    c1s = sorted(c1, key=lambda x: (x[0], x[1]))
    c2s = sorted(c2, key=lambda x: (x[0], x[1]))
    if c1s != c2s:
        return ResultDataType.FALSE
    if r1 != r2:
        return ResultDataType.TRUE

    return ResultDataType.TRUE


def compare_yaml(area_analyzer: AreaAnalyzer, entity_analyzer: EntityAnalyzer, property_analyzer: PropertyAnalyzer,
                 yaml_true_string,
                 yaml_pred_string) -> Result:
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
    is_area_exact_match = ResultDataType.FALSE
    are_entities_exactly_same = ResultDataType.FALSE
    are_entities_same_exclude_props = ResultDataType.FALSE
    are_relations_exactly_same = ResultDataType.NOT_APPLICABLE
    percentage_of_correctly_identified_properties = -1

    if generated_data:
        is_area_exact_match = area_analyzer.compare_areas_strict(ref_data['area'], generated_data['area'])
        is_area_light_match = area_analyzer.compare_areas_light(ref_data['area'], generated_data['area'])
        are_entities_exactly_same = entity_analyzer.compare_entities_strict(ref_data['entities'],
                                                                            generated_data['entities'])

        are_entities_same_exclude_props = entity_analyzer.compare_entities_exclude_props(ref_data['entities'],
                                                                                         generated_data['entities'])

        percentage_entities_partial_match_exclude_props = entity_analyzer.compare_entities_partial_match_exclude_props(
            ref_data['entities'],
            generated_data['entities'])

        # todo: property check
        num_entities_on_ref_data = len(ref_data['entities'])
        num_entities_on_gen_data = len(generated_data['entities'])

        ref_entities_with_properties = {}
        for ref_entity in ref_data['entities']:
            if 'properties' in ref_entity:
                ref_entities_with_properties[ref_entity['name']] = ref_entity['properties']

        if len(ref_entities_with_properties) > 0:
            # print("!!!entities with properties!!!!")
            predicted_entities_with_properties = {}
            for pred_entity in generated_data['entities']:
                if 'properties' in pred_entity:
                    predicted_entities_with_properties[pred_entity['name']] = pred_entity['properties']
            percentage_of_correctly_identified_properties = property_analyzer.percentage_of_correctly_identified_properties(
                ref_entities=ref_entities_with_properties, prop_entities=predicted_entities_with_properties)

        # todo: recheck this!!
        if 'relations' not in ref_data:
            are_relations_exactly_same = ResultDataType.NOT_APPLICABLE

        else:
            if 'relations' not in generated_data:
                are_relations_exactly_same = ResultDataType.FALSE

            else:
                are_relations_exactly_same = compare_relations(ref_data['relations'], generated_data['relations'])
    if (is_area_exact_match == ResultDataType.TRUE and are_entities_exactly_same == ResultDataType.TRUE
        and (are_relations_exactly_same == ResultDataType.TRUE or
             are_relations_exactly_same == ResultDataType.NOT_APPLICABLE)):
        is_perfect_match = ResultDataType.TRUE

    return Result(yaml_pred_string=yaml_pred_string,
                  yaml_true_string=yaml_true_string,
                  is_perfect_match=is_perfect_match,
                  is_parsable_yaml=_is_parsable_yaml,
                  is_area_exact_match=is_area_exact_match,
                  is_area_light_match=is_area_light_match,
                  num_entities_on_ref_data=num_entities_on_ref_data,
                  num_entities_on_gen_data=num_entities_on_gen_data,
                  are_entities_exactly_same=are_entities_exactly_same,
                  are_entities_same_exclude_props=are_entities_same_exclude_props,
                  percentage_entities_partial_match_exclude_props=percentage_entities_partial_match_exclude_props,
                  percentage_of_correctly_identified_properties=percentage_of_correctly_identified_properties,
                  are_relations_exactly_same=are_relations_exactly_same)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gold_file_path', type=str, required=True)
    parser.add_argument('--gold_sheet_name', type=str, required=True)
    parser.add_argument('--pred_file_path', type=str, required=True)
    parser.add_argument('--out_file_path', type=str, required=True)
    parser.add_argument('--out_file_path_sum', type=str, required=True)
    # parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    args = parser.parse_args()
    # geolocations_file_path = args.geolocations_file_path
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
    property_analyzer = PropertyAnalyzer()
    entity_analyzer = EntityAnalyzer(property_analyzer=property_analyzer)

    results = []
    for prediction, gold_label in tqdm(zip(predictions, gold_labels), total=len(gold_labels)):
        assert prediction['sentence'] == gold_label['sentence']
        yaml_pred_string = prediction['model_result']
        yaml_true_string = gold_label['YAML']
        result = compare_yaml(area_analyzer=area_analyzer,
                              entity_analyzer=entity_analyzer,
                              property_analyzer=property_analyzer,
                              yaml_true_string=yaml_true_string,
                              yaml_pred_string=yaml_pred_string)
        results.append(result.dict())

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

    evaluation_scores = {}

    # Results with binary type
    for result_type in ['is_perfect_match', 'is_parsable_yaml', 'is_area_exact_match',
                        'is_area_light_match',
                        'are_entities_exactly_same',
                        'are_entities_same_exclude_props',
                        'are_relations_exactly_same']:
        print(f"===Results for {result_type}===")
        na_samples = results[results[result_type] == ResultDataType.NOT_APPLICABLE]
        print(f"Number of NA samples: {len(na_samples)}")
        true_preds = results[results[result_type] == ResultDataType.TRUE]
        acc = len(true_preds) / (len(results) - len(na_samples))
        print(f'Accuracy of {result_type}')
        print(acc)

        evaluation_scores[result_type + "_NA"] = len(na_samples)
        evaluation_scores[result_type + "_acc"] = acc

    # Results with float type
    for result_type in ['percentage_entities_partial_match_exclude_props',
                        'percentage_of_correctly_identified_properties']:
        print(f"===Results for {result_type}===")

        na_results = results[results[result_type] == -1]
        valid_results = results[results[result_type] != -1]
        acc = np.mean(valid_results[result_type].to_numpy())
        print("Number of NA samples:", len(na_results))
        print(f'Average {result_type}')
        print(acc)

        evaluation_scores[result_type + "_NA"] = len(na_samples)
        evaluation_scores[result_type + "_acc"] = acc

    evaluation_scores = evaluation_scores | meta_results

    evaluation_scores = pd.DataFrame(evaluation_scores, index=[0])

    with pd.ExcelWriter(out_file_path) as writer:
        results.to_excel(writer)
    with pd.ExcelWriter(out_file_path_sum) as writer:
        evaluation_scores.to_excel(writer)
