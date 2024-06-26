import numpy as np
import pandas as pd
import random
import json
from itertools import combinations, product
from collections import defaultdict, Counter
from argparse import ArgumentParser

from benchmarking.add_values import add_values
from benchmarking.format_text import format_text


def generate_instructions(entities, areas, types, styles, typos, grammar_mistakes, relative_spatial_terms,
                          written_numbers, brand, multiple_of_one, names_or_areas_in_non_roman_alphabet,
                          min_items, max_items):  #

    optional_items_counts = {
        'typos_few': typos['few'],
        'typos_many': typos['many'],
        'grammar_few': grammar_mistakes['few'],
        'grammar_many': grammar_mistakes['many'],
        'spatial_yes': relative_spatial_terms['yes'],
        'number_yes': written_numbers['yes'],
        'brand_alone': brand['brand_alone'],
        'brand_type': brand['brand+type'],
        'multiple_of_one': multiple_of_one['yes'],
        'non_roman_area': names_or_areas_in_non_roman_alphabet['non_roman_area'],
        'non_roman_brand': names_or_areas_in_non_roman_alphabet['non_roman_brand']
    }

    required_combinations = list(product(entities, areas, types, styles))

    def generate_optional_combinations(nonzero_optional_num, zero_optional_num):
        buckets = [0, 0, 0, 0]
        all_combinations = []
        for r in range(min_items - 4, max_items - 4 + 1):
            for comb in combinations(optional_items_counts.keys(), r):
                if (("grammar_few" in comb and "grammar_many" in comb) or ("typos_few" in comb and "typos_many" in comb)
                        or ("brand_alone" in comb and "brand_type" in comb) or
                        ("non_roman_brand" in comb and not any(
                            item in comb for item in ["brand_alone", "brand_type"]))):
                    continue
                all_combinations.append(comb)
                buckets[r] += 1
        for r in range(min_items - 4 + 1, max_items - 4 + 1):
            combs = list(combinations(optional_items_counts.keys(), r))
            while nonzero_optional_num > buckets[r]:
                comb = combs[np.random.choice(np.arange(len(combs)))]
                if (("grammar_few" in comb and "grammar_many" in comb) or ("typos_few" in comb and "typos_many" in comb)
                        or ("brand_alone" in comb and "brand_type" in comb) or
                        ("non_roman_brand" in comb and not any(
                            item in comb for item in ["brand_alone", "brand_type"]))):
                    continue
                all_combinations.append(comb)
                buckets[r] += 1
        while zero_optional_num > buckets[0]:
            # combs = list(combinations(optional_items_counts.keys(), r))
            # comb = combs[np.random.choice(np.arange(len(combs)))]
            # if (("grammar_few" in comb and "grammar_many" in comb) or ("typos_few" in comb and "typos_many" in comb)
            #         or ("brand_alone" in comb and "brand_type" in comb) or
            #         ("non_roman_brand" in comb and not any(
            #             item in comb for item in ["brand_alone", "brand_type"]))):
            #     continue
            all_combinations.append(list(combinations(optional_items_counts.keys(), 0))[0])
            buckets[0] += 1

        random.shuffle(all_combinations)
        return all_combinations

    nonzero_optional_num = 150
    zero_optional_num = 50
    optional_combinations = generate_optional_combinations(nonzero_optional_num, zero_optional_num)

    item_counts = defaultdict(int)

    final_instructions = []

    entity_counter = 0
    area_counter = 0
    type_counter = 0
    style_counter = 0
    while not all(item_counts[item] == optional_items_counts[item] for item in optional_items_counts
                  if item != "non_roman_brand"):
        print(optional_items_counts, " -> ", item_counts)
        # entity, area, types, style = required_combinations[np.random.choice(np.arange(len(required_combinations)))]
        for comb in optional_combinations:
            def draw_vals():
                entity = entities[entity_counter]
                area = areas[area_counter]
                type = types[type_counter]
                style = styles[style_counter]

                return entity, area, type, style

            entity, area, type, style = draw_vals()
            while ((type in ["individual_distances",
                             "individual_distances_with_contains"] and entity != "3 Entities") or
                   (type in ["within_radius", "contains_relation"] and entity == "1 Entity") or
                   (type in ["within_radius", "in_area", "contains"] and "spatial_yes" in comb) or
                   (area == "No Area" and "non_roman_area" in comb)):

                if (type in ["within_radius", "in_area", "contains"] and "spatial_yes" in comb):
                    type_counter = (type_counter + 1) % len(types)
                elif (area == "No Area" and "non_roman_area" in comb):
                    area_counter = (area_counter + 1) % len(areas)
                else:
                    entity_counter = (entity_counter + 1) % len(entities)

                entity, area, type, style = draw_vals()

            if all(item_counts[item] < optional_items_counts[item] for item in comb):
                permutation = (entity, area, type, style) + comb
                final_instructions.append(permutation)
                for item in comb:
                    item_counts[item] += 1

            entity_counter = (entity_counter + 1) % len(entities)
            area_counter = (area_counter + 1) % len(areas)
            type_counter = (type_counter + 2) % len(types)
            style_counter = (style_counter + 1) % len(styles)

    for iid, instruction in enumerate(final_instructions):
        if (item_counts["non_roman_brand"] < optional_items_counts["non_roman_brand"] and
                len(instruction) < max_items and any(item in instruction for item in ["brand_alone", "brand_type"])
                and "non_roman_brand" not in instruction):
            final_instructions[iid] = instruction + ("non_roman_brand",)
            item_counts["non_roman_brand"] += 1

    random.shuffle(final_instructions)

    # if not all(item_counts[item] == optional_items_counts[item] for item in optional_items_counts):
    #     raise ValueError("Not all required counts are met")

    # valid_instructions = [inst for inst in final_instructions if min_items <= len(inst) <= max_items]

    return final_instructions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    parser.add_argument('--tag_combination_path', help='tag list file generated via retrieve_combinations')
    parser.add_argument('--tag_prop_examples_path', help='Examples of tag properties')
    parser.add_argument('--relative_spatial_terms_path', help='Path for the relative spats', required=True)
    args = parser.parse_args()

    tag_combination_path = args.tag_combination_path
    tag_prop_examples_path = args.tag_prop_examples_path
    geolocations_file_path = args.geolocations_file_path
    relative_spatial_terms_path = args.relative_spatial_terms_path

    # Input data
    entities = ["1 Entity", "2 Entities", "3 Entities"]  #3
    areas = ["No Area", "No Area", "City", "Region", "City, Country", "Region, Country", "City, Region, Country"]  #7
    types = ["in_area", "within_radius", "individual_distances", "contains_relation",
             "individual_distances_with_contains"]  #5
    styles = ["Simple language, all in one sentence.", "Simple language, multiple sentences.",
              "Elaborate wording, all in one sentence.", "Elaborate wording, multiple sentences.",
              "Short and precise, all in one sentence.", "Short and precise, multiple sentences."] #6
    # styles = ["in perfect grammar and clear wording", "in simple language",
    #           "with very precise wording, short, to the point", "with very elaborate wording",
    #           "as a chain of thoughts split into multiple sentences"]  #5

    typos = {"few": 50, "many": 50}
    grammar_mistakes = {"few": 50, "many": 50}
    relative_spatial_terms = {"yes": 100}
    written_numbers = {"yes": 100}
    brand = {"brand_alone": 50, "brand+type": 50}
    multiple_of_one = {"yes": 100}
    names_or_areas_in_non_roman_alphabet = {"non_roman_area": 50, "non_roman_brand": 50}
    # typos = {"few": 2, "many": 2}
    # grammar_mistakes = {"few": 2, "many": 2}
    # relative_spatial_terms = {"yes": 4}
    # written_numbers = {"yes": 4}
    # brand = {"brand_alone": 10, "brand+type": 10}
    # multiple_of_one = {"yes": 4}
    # names_or_areas_in_non_roman_alphabet = {"non_roman_area": 2, "non_roman_brand": 20}

    # add_props = {"1_property": 20, "2_properties": 20, "3_properties": 10}

    # Number of items in each permutation
    min_items = 4
    max_items = 7

    # Generate permutations
    instructions = generate_instructions(entities, areas, types, styles, typos, grammar_mistakes,
                                         relative_spatial_terms, written_numbers, brand, multiple_of_one,
                                         names_or_areas_in_non_roman_alphabet, min_items, max_items)  #

    # Print the result
    # for inst in instructions:
    #     print(inst[4:])

    # Check the counts
    count_result = Counter()
    for inst in instructions:
        count_result.update(inst)

    print()
    print(count_result)
    print(">>>", len(instructions))

    instructions_with_values = add_values(instructions, tag_combination_path,
                                          tag_prop_examples_path, geolocations_file_path, relative_spatial_terms_path)

    instructions_with_formatted_text = format_text(instructions_with_values)

    column_names = ["entities", "areas", "types", "styles", "optional 1", "optional 2", "optional 3", "optional 4",
                    "optional 5", "instructions"]
    # "area_instructions", "object_instructions", "relations_instructions",

    df = pd.DataFrame(instructions_with_formatted_text, columns=column_names)
    df.to_csv('benchmarking/results/instructions.csv', index=False, header=True)

    instructions_for_json = []
    with open("benchmarking/results/instructions.json", "w") as out_file:
        for id, inst in enumerate(instructions_with_formatted_text):
            json.dump({"id": "doc_" + str(id), "text": inst[-1]}, out_file)
            out_file.write('\n')

    print("\nOutput written to file!")
