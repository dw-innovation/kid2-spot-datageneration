import copy
import yaml
from argparse import ArgumentParser

def is_parsable_yaml(yaml_string):
    """
    Checks whether the input batch of YAML strings is parsable.

    :return: is_parsable, parsed_yaml - Boolean whether YAML is parsable plus parsed YAML (or None if not possible).
    """
    try:
        parsed_yaml = yaml.safe_load(yaml_string)
        is_parsable = True
    except:
        parsed_yaml = None
        is_parsable = False
    return is_parsable, parsed_yaml


def compare_areas(area1, area2):
    """
    Checks if two areas are identical.

    :param area1: The first area to compare.
    :param area2: The second area to compare.
    :return: Boolean whether the two areas are the same.
    """
    return area1 == area2


def compare_properties(props1, props2):
    """
    Check if two lists of properties are identical. The lists are first sorted via their names, to make sure the order
    does not affect the results.

    :param props1: The first property list to compare.
    :param props2: The second property list to compare.
    :return: Boolean whether the two property lists are the same.
    """
    if len(props1) != len(props2):
        return False
    props1_sorted = sorted(props1, key=lambda x: x['name'])
    props2_sorted = sorted(props2, key=lambda x: x['name'])
    return props1_sorted == props2_sorted


def compare_entities(entities1, entities2):
    """
    Check if two lists of entities are identical. The lists are first sorted via their names, to make sure the order
    does not affect the results.

    :param entities1: The first entity list to compare.
    :param entities2: The second entity list to compare.
    :return: Boolean whether the two entity lists are the same.
    """
    if len(entities1) != len(entities2):
        return False
    entities1_sorted = sorted(entities1, key=lambda x: x['name'])
    entities2_sorted = sorted(entities2, key=lambda x: x['name'])
    for ent1, ent2 in zip(entities1_sorted, entities2_sorted):
        if ent1['name'] != ent2['name'] or ent1['type'] != ent2['type']:
            return False
        if not compare_properties(ent1.get('properties', []), ent2.get('properties', [])):
            return False
    return True


def prepare_relation(data):
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
        prepped_relation[id]["source"] = [ent["name"] for ent in data["entities"] if ent["id"] == relations[id]["source"]][0]
        prepped_relation[id]["target"] = [ent["name"] for ent in data["entities"] if ent["id"] == relations[id]["target"]][0]
    return prepped_relation

def compare_relations(relations1, relations2):
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
        return False
    r1 = set()
    r2 = set()
    c1 = list()
    c2 = list()
    for id in range(len(relations1)):
        if relations1[id]["type"] == "contains":
            c1.append([relations1[id]["source"], relations1[id]["target"]])
        elif relations1[id]["type"] == "distance":
            r1.add(frozenset({relations1[id]["source"], relations1[id]["target"], relations1[id]["value"]}))
        if relations2[id]["type"] == "contains":
            c2.append([relations2[id]["source"], relations2[id]["target"]])
        elif relations2[id]["type"] == "distance":
            r2.add(frozenset({relations2[id]["source"], relations2[id]["target"], relations2[id]["value"]}))
    c1s = sorted(c1, key=lambda x: (x[0], x[1]))
    c2s = sorted(c2, key=lambda x: (x[0], x[1]))
    if c1s != c2s:
        return False
    if r1 != r2:
        return False

    return True

def compare_yaml(yaml_true_string, yaml_pred_string):
    """
    Compare two YAML structures represented as strings. This is done by comparing areas, entities and relations
    separately.

    :param yaml_true_string: The first YAML to compare.
    :param yaml_pred_string: The first YAML to compare.
    :return: Boolean whether the two YAMLs are the same.
    """
    _, data1 = is_parsable_yaml(yaml_true_string)
    is_parsable, data2 = is_parsable_yaml(yaml_pred_string)

    if not is_parsable:
        return False
    else:
        is_same = True
        if not compare_areas(data1['area'], data2['area']):
            print("XX Area is False!")
            is_same =  False
        else:
            print(">> Area is True!")

        if not compare_entities(data1['entities'], data2['entities']):
            print("XX Entities are False!")
            is_same =  False
        else:
            print(">> Entities are True!")

        if not compare_relations(prepare_relation(data1), prepare_relation(data2)): #['relations']
            print("XX Relations are False!")
            is_same =  False
        else:
            print(">> Relations are True!")

        return is_same

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    args = parser.parse_args()

    geolocations_file_path = args.geolocations_file_path

#     yaml_true_string = """area:
#   - type: area
#     value: Dâmbovița County, Romania
# entities:
#   - id: 0
#     name: social facility
#     type: nwr
#     properties:
#       - name: building levels
#         operator: <
#         value: 3
#       - name: name
#         operator: ~
#         value: ole
#   - id: 1
#     name: fabric shop
#     type: nwr
#   - id: 2
#     name: petrol station
#     type: nwr
# relations:
#   - type: distance
#     source: 1
#     target: 0
#     value: 400 m
#   - type: distance
#     source: 2
#     target: 1
#     value: 300 m"""
#
#     yaml_pred_string = """area:
#   - type: area
#     value: Dâmbovița County, Romania
# entities:
#   - id: 0
#     name: social facility
#     type: nwr
#     properties:
#       - name: building levels
#         operator: <
#         value: 3
#       - name: name
#         operator: ~
#         value: ole
#   - id: 1
#     name: petrol station
#     type: nwr
#   - id: 2
#     name: fabric shop
#     type: nwr
# relations:
#   - type: distance
#     source: 2
#     target: 0
#     value: 400 m
#   - type: distance
#     source: 1
#     target: 2
#     value: 300 m"""
    yaml_true_string = """area:
  - type: bbox
entities:
  - id: 0
    type: nwr
    name: vacant shop
    properties:
      - name: floors
        operator: <
        value: 10
  - id: 1
    type: nwr
    name: office building
  - id: 2
    type: nwr
    name: gambling den
relations:
  - type: contains
    source: 1
    target: 0 
  - type: distance
    source: 0
    target: 2
    value: 0.5 miles"""

    yaml_pred_string = """area:
  - type: bbox
entities:
  - id: 0
    type: nwr
    name: gambling den
  - id: 1
    type: nwr
    name: office building
  - id: 2
    type: nwr
    name: vacant shop
    properties:
      - name: floors
        operator: <
        value: 10
relations:
  - type: distance
    source: 0
    target: 2
    value: 0.5 miles
  - type: contains
    source: 1
    target: 2"""

    result = compare_yaml(yaml_true_string, yaml_pred_string)
    print("The YAML structures are the same:", result)
