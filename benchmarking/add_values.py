import numpy as np
import pandas as pd
import copy
import random

from datageneration.data_model import TagCombination, Relation, Relations, LocPoint, Property
from datageneration.generate_combination_table import QueryCombinationGenerator
from datageneration.area_generator import AreaGenerator, load_named_area_data, NamedAreaData
from datageneration.relation_generator import RelationGenerator
from datageneration.property_generator import PropertyGenerator
from datageneration.gpt_data_generator import PromptHelper, GPTDataGenerator, load_rel_spatial_terms


def add_values(instructions, tag_combination_path, tag_prop_examples_path, geolocations_file_path,
               relative_spatial_terms_path):
    tag_combinations = pd.read_json(tag_combination_path, lines=True).to_dict('records')
    tag_combinations = [TagCombination(**tag_comb) for tag_comb in tag_combinations]
    property_examples = pd.read_json(tag_prop_examples_path, lines=True).to_dict('records')

    def entnum_wrapper(num):
        def get_number_of_entities(max_number_of_entities_in_prompt):
            return num

        return get_number_of_entities

    query_comb_generator = QueryCombinationGenerator(geolocation_file=geolocations_file_path,
                                                          tag_combinations=tag_combinations,
                                                          property_examples=property_examples,
                                                          max_distance_digits=5,
                                                          percentage_of_two_word_areas=0.5,
                                                          prob_generating_contain_rel=0.3,
                                                          ratio_within_radius_within=0.5)
    area_generator = AreaGenerator(geolocation_file=geolocations_file_path, percentage_of_two_word_areas=0.5)
    relation_generator = RelationGenerator(max_distance_digits=5, prob_generating_contain_rel=0,
                                                    ratio_within_radius_within=0)
    property_generator = PropertyGenerator(property_examples)
    relative_terms = load_rel_spatial_terms(relative_spatial_terms_path)
    # gpt_generator = GPTDataGenerator(relative_terms, None, None, 0.3,
    #                                  0.3, 0, 5)
    gpt_generator = GPTDataGenerator(relative_terms, None, None, 0,
                                     0, 0, 5)
    prompt_helper = PromptHelper(relative_terms)

    add_properties = {"1_property": 15, "2_properties": 15, "3_properties": 5}
    add_properties = [id+1 for id, val in enumerate(add_properties.values()) for _ in range(val)]
    np.random.shuffle(add_properties)

    instructions_with_values = []
    for inst_id, instruction in enumerate(instructions):
        print("Generating ", inst_id+1, "/", len(instructions))
        max_within_combs = -1
        entities = None
        area = None
        relations = None
        area_entities = None
        point_entities = None
        # print(instruction)

        num_entities = int(instruction[0][0]) # Number of entities
        query_comb_generator.get_number_of_entities = entnum_wrapper(num_entities)

        # print(instruction[0], " -> ", query_comb_generator.get_number_of_entities(3))

        entities = query_comb_generator.generate_entities(3, 0,
                                                          0)

        if "brand_alone" in instruction:
            random_ent = np.random.choice(np.arange(len(entities)))
            entities[random_ent].is_area = False
            brand_examples = property_generator.select_named_property_example("brand~***example***")
            entities[random_ent].name = "brand: " + np.random.choice(brand_examples)

        while (instruction[2] in ["contains_relation", "individual_distances_with_contains"] and max_within_combs < 1):
            entities = query_comb_generator.generate_entities(3, 0,
                                                          0)
            if "brand_alone" in instruction:
                random_ent = np.random.choice(np.arange(len(entities)))
                entities[random_ent].is_area = False
                brand_examples = property_generator.select_named_property_example("brand~***example***")
                entities[random_ent].name = "brand: " + np.random.choice(brand_examples)

            area_entities = []
            point_entities = []
            for id, entity in enumerate(entities):
                if entity.is_area:
                    area_entities.append(entity)
                else:
                    point_entities.append(entity)

            if instruction[2] == "contains_relation" and len(area_entities) > 1:
                continue

            max_within_combs = min(len(area_entities), len(point_entities))

        brand_drawn = False
        for id, e in enumerate(entities):
            if e.name in ["brand", "brand name", "name"]:
                if "brand_alone" not in instruction:
                    instruction = instruction + ("brand_alone",)
                continue
            for tc in tag_combinations:
                if e.name in tc.descriptors:
                    if len(add_properties) > 0 and np.random.choice([True, False], p=[0.4,0.6]):
                        num_props = add_properties.pop(-1)
                        if len(tc.tag_properties) > num_props:
                            # drawn_props = np.random.sample(tc.tag_properties,num_props)
                            drawn_props = query_comb_generator.generate_properties(candidate_properties=tc.tag_properties,
                                                 num_of_props=num_props)
                            entities[id].properties = drawn_props
                            if "with_properties" not in instruction:
                                instruction = instruction + ("with_properties",)
                            for dp in drawn_props:
                                if dp.name in ["brand", "band name", "name"]:
                                    if not "brand_type" in instruction:
                                        instruction = instruction + ("brand_type",)
                                    brand_drawn = True
                        else:
                            add_properties.append(num_props)

                    if "brand" in tc.descriptors and "brand_type" in instruction and not brand_drawn:
                        brand_ex = property_generator.select_named_property_example("brand~***example***")
                        brand_prop = Property(name='brand name', operator='~', value=brand_ex)
                        if entities[id].properties:
                            entities[id].properties = entities[id].properties.append(brand_prop)
                        else:
                            entities[id].properties = [brand_prop]
                        if "with_properties" not in instruction:
                            instruction = instruction + ("with_properties",)

        if "brand_type" in instruction and not brand_drawn:
            instruction = tuple(x for x in instruction if x != "brand_type")
            if "non_roman_brand" in instruction:
                instruction = tuple(x for x in instruction if x != "non_roman_brand")

        if instruction[2] == "in_area":
            relations = relation_generator.generate_in_area(len(entities))
            relations = Relations(type=instruction[2], relations=relations)
        elif instruction[2] == "within_radius":
            relations = relation_generator.generate_within_radius(num_entities)
            relations = Relations(type=instruction[2], relations=relations)
        elif instruction[2] == "individual_distances":
            relations = relation_generator.generate_individual_distances([e.id for e in entities])
            relations = Relations(type=instruction[2], relations=relations)
        elif instruction[2] in ["contains_relation", "individual_distances_with_contains"]:
            relations_type = None
            while relations_type != instruction[2]:
                relations = relation_generator.generate_relation_with_contain(area_entities, point_entities, 1)
                relations_type = relations.type
            entities, relations = query_comb_generator.sort_entities(entities, relations)

        instruct_relations = copy.deepcopy(relations)

        def update_relation_distance(relations, relation_to_be_updated, distance):
            updated_relations = []
            return_relations = copy.deepcopy(relations)
            for relation in return_relations.relations:
                if relation == relation_to_be_updated:
                    relation.value = distance
            return return_relations

        area_instructions = ""
        object_instructions = ""
        relations_instructions = ""
        relspat_added = False
        writtendist_added = False
        prob_usage_of_relative_spatial_terms = 0
        prob_usage_of_written_numbers = 0
        if "spatial_yes" in instruction:
            prob_usage_of_relative_spatial_terms = 1.0
        if "number_yes" in instruction:
            prob_usage_of_written_numbers = 1.0

        relation_prompts = []
        if relations.relations:
            if "contains" in relations.type:
                generated_prompt, _ = prompt_helper.add_relation_with_contain(
                    relations.relations, entities)
                for p in generated_prompt.split("\n"):
                    relation_prompts.append(p + "\n")
                relation_prompts[-1] = relation_prompts[-1][:-1]
            for rel_id, relation in enumerate(relations.relations):
                if relation.type != "contains":
                    use_relative_spatial_terms = np.random.choice([False, True], p=[
                        1.0 - prob_usage_of_relative_spatial_terms, prob_usage_of_relative_spatial_terms])
                    use_written_distance = np.random.choice([False, True], p=[
                        1.0 - prob_usage_of_written_numbers, prob_usage_of_written_numbers])
                    # In case both relative term and written word are selected, randomly only select one of them
                    if use_relative_spatial_terms and use_written_distance:
                        if random.choice([True, False]):
                            use_relative_spatial_terms = False
                        else:
                            use_written_distance = False
                    if use_relative_spatial_terms:
                        generated_prompt, overwritten_distance = prompt_helper.add_relative_spatial_terms(relation,
                                                                                                          entities)
                        relation_prompts.append(generated_prompt)
                        raw_term = generated_prompt.split(": ")[1][:-1]

                        relations = update_relation_distance(relations=relations,
                                                      relation_to_be_updated=relation,
                                                      distance=overwritten_distance)
                        instruct_relations = update_relation_distance(relations=instruct_relations,
                                                      relation_to_be_updated=relation,
                                                      distance=overwritten_distance)

                        if prob_usage_of_written_numbers == 1.0:
                            prob_usage_of_relative_spatial_terms = 0.0
                        else:
                            prob_usage_of_relative_spatial_terms = 0.3
                        if prob_usage_of_written_numbers == 0.0:
                            prob_usage_of_written_numbers = 0.3

                        relspat_added = True
                    elif use_written_distance:
                        metric = relation.value.split()[-1]
                        numeric_distance, written_distance = prompt_helper.generate_written_word_distance(
                            metric, gpt_generator.max_dist_digits)
                        # written_distance_relation = Relation(type=relation.type, source=relation.source,
                        #                                      target=relation.target, value=written_distance)

                        relations = update_relation_distance(relations=relations,
                                                      relation_to_be_updated=relation,
                                                      distance=numeric_distance)
                        instruct_relations = update_relation_distance(relations=instruct_relations,
                                                      relation_to_be_updated=relation,
                                                      distance=written_distance)

                        generated_prompt = prompt_helper.add_desc_away_prompt(instruct_relations.relations[rel_id], entities)
                        relation_prompts.append(generated_prompt)

                        if prob_usage_of_relative_spatial_terms == 1.0:
                            prob_usage_of_written_numbers = 0.0
                        else:
                            prob_usage_of_written_numbers = 0.3
                        if prob_usage_of_relative_spatial_terms == 0.0:
                            prob_usage_of_relative_spatial_terms = 0.3

                        writtendist_added = True

                    else:
                        generated_prompt = prompt_helper.add_desc_away_prompt(relations.relations[rel_id], entities)
                        relation_prompts.append(generated_prompt)

        if "spatial_yes" in instruction and not relspat_added:
            instruction = tuple(x for x in instruction if x != "spatial_yes")
        if "number_yes" in instruction and not writtendist_added:
            instruction = tuple(x for x in instruction if x != "number_yes")

        # print(instruction[2], " -> ", relations)

        if instruction[1] == "No Area":
            area = area_generator.generate_no_area()
        elif instruction[1] == "City":
            area = area_generator.generate_city_area()
        elif instruction[1] == "Region":
            area = area_generator.generate_region_area()
        elif instruction[1] == "City, Country":
            area = area_generator.generate_city_and_country_area()
        elif instruction[1] == "Region, Country":
            area = area_generator.generate_region_and_country_area()
        elif instruction[1] == "City, Region, Country":
            area = area_generator.generate_city_and_region_and_country_area()

        padded_instruction = list(instruction) + [""] * (9 - len(instruction))

        query_yaml = LocPoint(area=area, entities=entities, relations=relations)
        instruct_yaml = LocPoint(area=area, entities=entities, relations=instruct_relations)
        _, prompt = gpt_generator.generate_prompt(instruct_yaml, "", "")

        object_instructions = prompt.split("Objects:")[1]
        object_instructions = object_instructions.split("\nPlease take your time")[0]
        if "Relations:\n" in object_instructions:
            if relations.type in ["individual_distances", "individual_distances_with_contains"]:
                for rp in relation_prompts:
                    relations_instructions = relations_instructions + rp
            else:
                relations_instructions = object_instructions.split("Relations:\n")[1]
            object_instructions = object_instructions.split("Relations:")[0]
            # elif relations.type == "within_radius":
            #     relations_instructions += gpt_generator.radius_prompt_generation(instruct_relations)
        if area.value:
            area_instructions = area.value

        # print(query_instructions)

        instructions_with_values.append(list(padded_instruction) + [area_instructions.rstrip(),
                                    object_instructions.rstrip(), relations_instructions.rstrip(), query_yaml])


        # print(list(padded_instruction) + [LocPoint(area=area, entities=entities, relations=relations)])

    return instructions_with_values