import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from typing import List

from datageneration.area_generator import AreaGenerator, NamedAreaData, load_named_area_data
from datageneration.data_model import TagAttributeExample, TagAttribute, Property, TagCombination, Entity, Relations, \
    LocPoint, Area
from datageneration.property_generator import PropertyGenerator
from datageneration.relation_generator import RelationGenerator
from datageneration.utils import write_output


class QueryCombinationGenerator(object):
    def __init__(self, geolocations: List[NamedAreaData], tag_combinations: List[TagCombination],
                 attribute_examples: List[TagAttributeExample], max_distance_digits: int):
        self.entity_tag_combinations = list(filter(lambda x: 'core' in x.comb_type.value, tag_combinations))
        self.area_generator = AreaGenerator(geolocations)
        self.property_generator = PropertyGenerator(attribute_examples)
        self.relation_generator = RelationGenerator(max_distance_digits=max_distance_digits)

    def index_to_descriptors(self, index):
        return self.all_tags[int(index)]['descriptors']

    def get_number_of_entities(self, max_number_of_entities_in_prompt: int) -> int:
        """
        This method of selecting the number of entities uses an exponential decay method that returns
        a probability distribution that has a peak probability value, from which the probabilities decrease
        towards both sides. The decay rate can be customised per side to allow for the selection of a higher
        decay rate towards the left to minimise one and two entity samples. This is due to the fact that these
        queries don't have sufficient entities to assign all three query types and should hence occur less in the
        training data.

        Example probability distribution with 4 ents, peak 3, decay left 0.3, decay right 0.4:
            [0.0993, 0.1999, 0.4026, 0.2982]

        :param max_number_of_entities_in_prompt: The maximum allowed number of entities per query
        :return: The selected number of entities
        """
        peak_value = 3  # Number of entity with the highest probability
        decay_rate_right = 0.7
        decay_rate_left = 0.3
        entity_nums = np.arange(1, max_number_of_entities_in_prompt + 1)
        probabilities = np.zeros(max_number_of_entities_in_prompt)
        probabilities[peak_value - 1] = 1
        probabilities[peak_value:] = np.exp(-decay_rate_right * (entity_nums[peak_value:] - peak_value))
        probabilities[:peak_value] = np.exp(-decay_rate_left * (peak_value - entity_nums[:peak_value]))
        probabilities /= np.sum(probabilities)
        number_of_entities_in_prompt = np.random.choice(entity_nums, p=probabilities)

        return number_of_entities_in_prompt

    def get_number_of_props(self, max_number_of_props_in_entity: int):
        """
        This method of selecting the number of properties uses an exponential decay method that returns
        a probability distribution that assigns higher probabilities to lower values, as many entities
        with multiple properties will result in convoluted sentence

        Example probability distribution with 4 props & decay of 0.3: [0.3709, 0.2748, 0.2036, 0.1508]

        :param max_number_of_props_in_entity: The maximum allowed number of properties per entity
        :return: The selected number of properties
        """
        decay_rate = 0.3
        prop_nums = np.arange(1, max_number_of_props_in_entity + 1)
        probabilities = np.exp(-decay_rate * prop_nums)
        probabilities /= np.sum(probabilities)
        selected_num_of_props = np.random.choice(prop_nums, p=probabilities)

        return selected_num_of_props

    def generate_entities(self, max_number_of_entities_in_prompt: int, max_number_of_props_in_entity: int,
                          percentage_of_entities_with_props: float) -> List[Entity]:
        """
        Generates a list of entities with associated properties based on random selection of descriptors.

        Args:
            max_number_of_entities_in_prompt (int): Number of entities to generate.
            max_number_of_props_in_entity (int): Maximum number of properties each entity can have.
            percentage_of_entities_with_props (float): Ratio of entities that have a non-zero number of properties

        Returns:
            List[Entity]: A list of generated entities with associated properties.

        Note:
            - The function selects a random subset of descriptors from the available combinations of entity tag combinations.
            - Each entity is assigned a random name chosen from the selected descriptors.
            - If `max_number_of_props_in_entity` is greater than or equal to 1, properties are generated for each entity.
              Otherwise, entities are generated without properties.
        """
        number_of_entities_in_prompt = self.get_number_of_entities(max_number_of_entities_in_prompt)

        selected_entities = []
        selected_entity_numbers = []
        while len(selected_entities) < number_of_entities_in_prompt:
            selected_idx_for_combinations = np.random.randint(0, len(self.entity_tag_combinations))
            if selected_idx_for_combinations in selected_entity_numbers:
                continue
            selected_entity_numbers.append(selected_idx_for_combinations)
            selected_tag_comb = self.entity_tag_combinations[selected_idx_for_combinations]
            associated_descriptors = selected_tag_comb.descriptors
            entity_name = np.random.choice(associated_descriptors)
            is_area = selected_tag_comb.is_area

            # Randomise whether probabilities should be added to ensure high enough ratio of zero property cases
            add_properties = np.random.choice([True, False], p=[percentage_of_entities_with_props,
                                                                1 - percentage_of_entities_with_props])
            if add_properties and max_number_of_props_in_entity >= 1:
                candidate_attributes = selected_tag_comb.tag_attributes
                if len(candidate_attributes) == 0:
                    continue
                current_max_number_of_props = min(len(candidate_attributes), max_number_of_props_in_entity)
                if current_max_number_of_props > 1:
                    # selected_num_of_props = np.random.randint(1, max_number_of_props_in_entity)
                    selected_num_of_props = self.get_number_of_props(current_max_number_of_props)
                else:
                    selected_num_of_props = current_max_number_of_props
                properties = self.generate_properties(candidate_attributes=candidate_attributes,
                                                      num_of_props=selected_num_of_props)
                selected_entities.append(
                    Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=properties))
            else:
                selected_entities.append(
                    Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=[]))

        return selected_entities

    def generate_properties(self, candidate_attributes: List[TagAttribute], num_of_props: int) -> List[Property]:
        candidate_indices = np.arange(len(candidate_attributes))
        np.random.shuffle(candidate_indices)
        selected_indices = candidate_indices[:num_of_props]
        tag_properties = []
        for idx in selected_indices:
            tag_attribute = candidate_attributes[idx]
            tag_property = self.property_generator.run(tag_attribute)
            tag_properties.append(tag_property)

        return tag_properties

    # todo make it independent from entities
    def generate_relations(self, entities: List[Entity]) -> Relations:
        relations = self.relation_generator.run(entities=entities)
        return relations

    def run(self, num_queries: int, max_number_of_entities_in_prompt: int, max_number_of_props_in_entity: int,
            percentage_of_entities_with_props: float) -> List[LocPoint]:
        '''
        A method that generates random query combinations and optionally saves them to a JSON file.
        It gets a list of random tag combinations and adds additional information that is required to generate
        full queries, including area names, and different search tasks.
        The current search tasks are: (1) individual distances: a random specific distance is defined between all objects,
        (2) within radius: a single radius within which all objects are located, (3) in area: general search for all objects
        within given area.

        TODO: Write/update all docstrings, maybe use this text somewhere else

        :param num_queries: (int) TODO
        :param max_number_of_entities_in_prompt: (int) TODO
        :param max_number_of_props_in_entity: (int) TODO
        :param percentage_of_entities_with_props: (int) TODO
        :param percentage_of_entities_with_props: (float) TODO
        :return: loc_points (List[LocPoint])
        '''
        # ipek - node types are not used
        node_types = ["nwr", "cluster", "group"]
        loc_points = []
        for _ in tqdm(range(num_queries), total=num_queries):
            entities = self.generate_entities(max_number_of_entities_in_prompt=max_number_of_entities_in_prompt,
                                              max_number_of_props_in_entity=max_number_of_props_in_entity,
                                              percentage_of_entities_with_props=percentage_of_entities_with_props)
            area = self.generate_area()
            relations = self.generate_relations(entities=entities)

            loc_points.append(LocPoint(area=area, entities=entities, relations=relations))

        return loc_points

    def generate_area(self) -> Area:
        return self.area_generator.run()


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    parser.add_argument('--tag_combination_path', help='tag list file generated via retrieve_combinations')
    parser.add_argument('--tag_attribute_examples_path', help='Examples of tag attributes')
    parser.add_argument('--output_file', help='File to save the output')
    parser.add_argument('--max_distance_digits', help='Define max distance', type=int)
    parser.add_argument('--write_output', action='store_true')
    parser.add_argument('--samples', help='Number of the samples to generate', type=int)
    parser.add_argument('--max_number_of_entities_in_prompt', type=int, default=4)
    parser.add_argument('--max_number_of_props_in_entity', type=int, default=4)
    parser.add_argument('--percentage_of_entities_with_props', type=float, default=0.3)

    args = parser.parse_args()

    tag_combination_path = args.tag_combination_path
    tag_attribute_examples_path = args.tag_attribute_examples_path
    geolocations_file_path = args.geolocations_file_path
    max_distance_digits = args.max_distance_digits
    num_samples = args.samples
    output_file = args.output_file
    max_number_of_entities_in_prompt = args.max_number_of_entities_in_prompt
    max_number_of_props_in_entity = args.max_number_of_props_in_entity
    percentage_of_entities_with_props = args.percentage_of_entities_with_props

    tag_combinations = pd.read_json(tag_combination_path, lines=True).to_dict('records')
    tag_combinations = [TagCombination(**tag_comb) for tag_comb in tag_combinations]

    attribute_examples = pd.read_json(tag_attribute_examples_path, lines=True).to_dict('records')
    geolocations = load_named_area_data(geolocations_file_path)

    query_comb_generator = QueryCombinationGenerator(geolocations=geolocations,
                                                     tag_combinations=tag_combinations,
                                                     attribute_examples=attribute_examples,
                                                     max_distance_digits=args.max_distance_digits)

    generated_combs = query_comb_generator.run(num_queries=num_samples,
                                               max_number_of_entities_in_prompt=max_number_of_entities_in_prompt,
                                               max_number_of_props_in_entity=max_number_of_props_in_entity,
                                               percentage_of_entities_with_props=percentage_of_entities_with_props)

    if args.write_output:
        write_output(generated_combs, output_file=output_file)
