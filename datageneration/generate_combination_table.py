import copy
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from datageneration.area_generator import AreaGenerator, NamedAreaData, load_named_area_data
from datageneration.data_model import TagPropertyExample, TagProperty, Property, TagCombination, Entity, Relations, \
    LocPoint, Area
from datageneration.property_generator import PropertyGenerator, fetch_color_bundle
from datageneration.relation_generator import RelationGenerator
from datageneration.utils import get_random_decimal_with_metric, write_output

"""
Query combination generator

This module assembles full query objects by sampling:
- An area (with optional non‑Roman script variants).
- One or more entities (optionally with properties).
- Inter-entity relations (distance / containment).

The main entry point is `QueryCombinationGenerator.run`, which returns a list of
`LocPoint` objects. A CLI is provided for batch generation and optional writing
to disk.
"""

class QueryCombinationGenerator(object):
    """Compose random query combinations (area, entities, relations, properties).

    This orchestrates:
      - Area generation (`AreaGenerator`)
      - Entity selection with descriptors/tag combinations
      - Property sampling per entity (`PropertyGenerator`)
      - Relation generation (`RelationGenerator`)
      - Optional clustering attributes for entities

    Sampling is controlled by the probability parameters provided at init.

    Args:
        geolocation_file: Path to the geo DB JSON (countries > states > cities).
        non_roman_vocab_file: Path to non‑Roman name vocabulary JSON.
        tag_combinations: Allowed entity/tag combinations with descriptors & props.
        property_examples: Raw property example rows (JSON lines → dicts).
        max_distance_digits: Max number of digits for generated distance magnitudes.
        prob_of_two_word_areas: P(area has two+ words) when sampling areas.
        prob_generating_contain_rel: P(to generate containment relations).
        prob_adding_brand_names_as_entity: P(to inject a brand:* entity).
        prob_of_numerical_properties: P(category weight for numerical props).
        prob_of_color_properties: P(category weight for color props).
        prob_of_popular_non_numerical_properties: P(weight for popular non-num props).
        prob_of_other_non_numerical_properties: P(weight for other non-num props).
        prob_of_rare_non_numerical_properties: P(weight for rare non-num props).
        prob_of_non_roman_areas: P(to translate area to non‑Roman script if possible).
        color_bundle_path: Path to color bundles used by `PropertyGenerator`.
        prob_of_cluster_entities: P(to convert an entity to type='cluster' with
            `minPoints` and `maxDistance` set).
    """
    def __init__(self, geolocation_file: str,
                 non_roman_vocab_file: str,
                 tag_combinations: List[TagCombination],
                 property_examples: List[TagPropertyExample],
                 max_distance_digits: int,
                 prob_of_two_word_areas: float,
                 prob_generating_contain_rel: float,
                 prob_adding_brand_names_as_entity: float,
                 prob_of_numerical_properties: float,
                 prob_of_color_properties: float,
                 prob_of_popular_non_numerical_properties: float,
                 prob_of_other_non_numerical_properties: float,
                 prob_of_rare_non_numerical_properties: float,
                 prob_of_non_roman_areas: float,
                 color_bundle_path: str,
                 prob_of_cluster_entities: float
                 ):

        color_bundles = fetch_color_bundle(property_examples=property_examples,bundle_path=color_bundle_path)
        self.property_generator = PropertyGenerator(property_examples, color_bundles=color_bundles)
        self.entity_tag_combinations = self.categorize_entities_based_on_their_props(list(filter(lambda x: 'core' in x.comb_type.value, tag_combinations)))

        self.area_generator = AreaGenerator(geolocation_file=geolocation_file, non_roman_vocab_file=non_roman_vocab_file, prob_of_two_word_areas=prob_of_two_word_areas, prob_of_non_roman_areas=prob_of_non_roman_areas)
        self.prob_adding_brand_names_as_entity = prob_adding_brand_names_as_entity
        self.relation_generator = RelationGenerator(max_distance_digits=max_distance_digits,
                                                    prob_generating_contain_rel=prob_generating_contain_rel)
        self.prob_of_numerical_properties = prob_of_numerical_properties
        self.prob_of_color_properties = prob_of_color_properties
        self.prob_of_popular_non_numerical_properties = prob_of_popular_non_numerical_properties
        self.prob_of_other_non_numerical_properties = prob_of_other_non_numerical_properties
        self.prob_of_rare_non_numerical_properties = prob_of_rare_non_numerical_properties
        self.prob_of_cluster_entities = prob_of_cluster_entities
        self.all_properties_with_probs = {
            "numerical": self.prob_of_numerical_properties,
            "colour": self.prob_of_color_properties,
            'rare_non_numerical': self.prob_of_rare_non_numerical_properties,
            "popular_non_numerical": self.prob_of_popular_non_numerical_properties,
            "other_non_numerical": self.prob_of_other_non_numerical_properties,
        }

    def categorize_entities_based_on_their_props(self, tag_combinations: List[TagCombination]) -> Dict:
        """Bucket tag combinations by property category for targeted sampling.

        Args:
            tag_combinations: List of `TagCombination` (typically core entities).

        Returns:
            Dict mapping category → list of `TagCombination`, with keys:
                'numerical', 'colour', 'rare_non_numerical', 'popular_non_numerical',
                'other_non_numerical', and 'default' (all core combos).
        """
        categorized_entities = {
            'numerical': [],
            'colour': [],
            'rare_non_numerical': [],
            'popular_non_numerical': [],
            'other_non_numerical': [],
            'default': [] # add every type of entities here
        }
        for tag_combination in tag_combinations:
            tag_properties = tag_combination.tag_properties
            categorized_entities['default'].append(tag_combination)
            prop_categories = self.property_generator.categorize_properties(tag_properties=tag_properties)
            for prop_key in prop_categories.keys():
                categorized_entities[prop_key].append(tag_combination)
        return categorized_entities

    def get_number_of_entities(self, max_number_of_entities_in_prompt: int) -> int:
        """Sample number of properties per entity favoring smaller counts.

        Uses an exponential decay to keep sentences readable.

        Example probability distribution with 4 ents, peak 3, decay left 0.3, decay right 0.4:
            [0.0993, 0.1999, 0.4026, 0.2982]

        Args:
            max_number_of_props_in_entity: Upper bound of properties per entity.

        Returns:
            The sampled number of properties (int) in [1, max_number_of_props_in_entity].
        """
        peak_value = 3  # Number of entity with the highest probability
        decay_rate_right = 0.7
        decay_rate_left = 0.6 #0.3
        entity_nums = np.arange(1, max_number_of_entities_in_prompt + 1)
        probabilities = np.zeros(max_number_of_entities_in_prompt)
        probabilities[peak_value - 1] = 1
        probabilities[peak_value:] = np.exp(-decay_rate_right * (entity_nums[peak_value:] - peak_value))
        probabilities[:peak_value] = np.exp(-decay_rate_left * (peak_value - entity_nums[:peak_value]))
        probabilities /= np.sum(probabilities)
        number_of_entities_in_prompt = np.random.choice(entity_nums, p=probabilities)

        return number_of_entities_in_prompt

    def get_number_of_props(self, max_number_of_props_in_entity: int):
        """Sample number of properties per entity favoring smaller counts.

        Uses an exponential decay to keep sentences readable.

        Example probability distribution with 4 props & decay of 0.3: [0.3709, 0.2748, 0.2036, 0.1508]

        Args:
            max_number_of_props_in_entity: Upper bound of properties per entity.

        Returns:
            The sampled number of properties (int) in [1, max_number_of_props_in_entity].
        """
        decay_rate = 0.5
        prop_nums = np.arange(1, max_number_of_props_in_entity + 1)
        probabilities = np.exp(-decay_rate * prop_nums)
        probabilities /= np.sum(probabilities)
        selected_num_of_props = np.random.choice(prop_nums, p=probabilities)

        return selected_num_of_props

    def add_cluster_entities(self, selected_entities)
        """Optionally convert entities to 'cluster' type with minPoints/maxDistance.

        Args:
            selected_entities: Entities to augment.

        Returns:
            Updated list with some entities converted to clusters.
        """
        for id, entity in enumerate(selected_entities):
            add_cluster = np.random.choice([True, False], p=[self.prob_of_cluster_entities,
                                                                     1 - self.prob_of_cluster_entities])
            if add_cluster:
                minPoints = np.random.choice(np.arange(20))
                maxDistance = get_random_decimal_with_metric(5)
                selected_entities[id] = Entity(id=selected_entities[id].id, is_area=selected_entities[id].is_area,
                                                name=selected_entities[id].name, type='cluster',
                                                minPoints=minPoints, maxDistance=maxDistance, properties=selected_entities[id].properties)

        return selected_entities

    def generate_entities(self, max_number_of_entities_in_prompt: int, max_number_of_props_in_entity: int,
                          prob_of_entities_with_props: float) -> List[Entity]:
        """Generate entities (optionally with properties) from tag combinations.

        Sampling strategy:
          1) Decide number of entities.
          2) For each, optionally add properties based on category weights.
          3) Occasionally inject a brand:* entity.
          4) Optionally convert to cluster.

        Args:
            max_number_of_entities_in_prompt: Max entities to include in the query.
            max_number_of_props_in_entity: Max properties allowed per entity.
            prob_of_entities_with_props: Probability an entity carries ≥1 property.

        Returns:
            List of `Entity` objects.
        """
        number_of_entities_in_prompt = self.get_number_of_entities(max_number_of_entities_in_prompt)
        selected_entities = []
        selected_tag_combs = []
        while len(selected_entities) < number_of_entities_in_prompt:
            selected_brand_name = np.random.choice([True, False], p=[self.prob_adding_brand_names_as_entity,
                                                                     1 - self.prob_adding_brand_names_as_entity])
            if not selected_brand_name:
                add_properties = np.random.choice([True, False], p=[prob_of_entities_with_props,
                                                                1 - prob_of_entities_with_props])
                if add_properties and max_number_of_props_in_entity >= 1:
                    selected_property_category = np.random.choice(list(self.all_properties_with_probs.keys()),
                                                                  p=list(self.all_properties_with_probs.values()))

                    selected_idx_for_combinations = np.random.randint(0, len(self.entity_tag_combinations[selected_property_category]))
                    selected_tag_comb = self.entity_tag_combinations[selected_property_category][selected_idx_for_combinations]
                    is_area = selected_tag_comb.is_area

                    if selected_tag_comb in selected_tag_combs:
                        continue

                    selected_tag_combs.append(selected_tag_comb)
                    associated_descriptors = selected_tag_comb.descriptors

                    entity_name = np.random.choice(associated_descriptors)
                    candidate_properties = selected_tag_comb.tag_properties

                    if len(candidate_properties) == 0:
                        continue

                    current_max_number_of_props = min(len(candidate_properties), max_number_of_props_in_entity)
                    if current_max_number_of_props > 1:
                        selected_num_of_props = self.get_number_of_props(current_max_number_of_props)
                    else:
                        selected_num_of_props = current_max_number_of_props

                    properties = self.generate_properties(candidate_properties=candidate_properties,
                                                          num_of_props=selected_num_of_props)
                    selected_entities.append(
                        Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=properties))
                else:
                    selected_idx_for_combinations = np.random.randint(0, len(self.entity_tag_combinations['default']))
                    selected_tag_comb = self.entity_tag_combinations['default'][selected_idx_for_combinations]
                    associated_descriptors = selected_tag_comb.descriptors
                    entity_name = np.random.choice(associated_descriptors)

                    is_area = selected_tag_comb.is_area
                    if selected_tag_comb in selected_tag_combs:
                        continue
                    selected_tag_combs.append(selected_tag_comb)
                    selected_entities.append(
                        Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=[]))
            else:
                brand_examples = self.property_generator.select_named_property_example("brand~***example***")
                entity_name = f"brand:{np.random.choice(brand_examples)}"
                is_area = False
                selected_entities.append(
                    Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=[]))

        selected_entities = self.add_cluster_entities(selected_entities)

        return selected_entities

    def generate_properties(self, candidate_properties: List[TagProperty], num_of_props: int, trial_err_count=100) -> List[Property]:
        """Select and instantiate a set of properties from candidates.

        The method:
          - Filters property categories based on configured probabilities.
          - Samples categories and then specific tag properties.
          - Ensures no duplicate descriptor keys within the same entity.

        Args:
            candidate_properties: Tag properties available for the chosen entity.
            num_of_props: Number of properties to generate.
            trial_err_count: Max attempts before giving up (prevents infinite loops).

        Returns:
            A list of concrete `Property` instances.
        """
        categorized_properties = self.property_generator.categorize_properties(candidate_properties)
        all_property_categories = list(self.all_properties_with_probs.keys())
        all_properties_with_probs = self.all_properties_with_probs

        new_all_property_categories = [
            category for category in all_property_categories
            if all_properties_with_probs.get(category) != 0.0 and category in categorized_properties
        ]
        new_all_property_category_probs = {
            category: prob for category, prob in all_properties_with_probs.items()
            if prob != 0.0 and category in categorized_properties
        }

        all_property_categories = new_all_property_categories
        all_property_category_probs = new_all_property_category_probs

        all_property_category_probs_values = list(all_property_category_probs.values())
        tag_properties = []
        tag_properties_keys = []

        trial_err = 0
        while(len(tag_properties)<num_of_props):
            if trial_err == trial_err_count:
                return tag_properties
            trial_err += 1
            if sum(all_property_category_probs_values) != 1:
                remaining_prob = (1- sum(all_property_category_probs_values)) / len(all_property_category_probs_values)
                all_property_category_probs_values = list(map(lambda x: x+remaining_prob, all_property_category_probs_values))

            selected_property_category = np.random.choice(all_property_categories, p=all_property_category_probs_values)
            selected_category_properties = categorized_properties[selected_property_category]
            candidate_indices = np.arange(len(selected_category_properties))
            np.random.shuffle(candidate_indices)
            selected_index = candidate_indices[0]
            tag_property = selected_category_properties[selected_index]
            tag_props_key = ' '.join(tag_property.descriptors)
            # if tag_props_key in tag_properties_keys and tag_props_key not in ['cuisine', 'sport']:
                # we keep cuisine, sport because facilities can serve multiple cuisine, and offer different sport activities
                # continue
            if tag_props_key not in tag_properties_keys: # Ensure no duplicates
                tag_properties_keys.append(tag_props_key)
                tag_property = self.property_generator.run(tag_property)
                tag_properties.append(tag_property)

        return tag_properties

    # todo make it independent from entities
    def generate_relations(self, entities: List[Entity]) -> Relations:
        """Generate relations between entities via the `RelationGenerator`.

        Args:
            entities: Entities to consider when creating relations.

        Returns:
            A `Relations` object describing distances/containment.
        """
        relations = self.relation_generator.run(entities=entities)
        return relations

    def sort_entities(self, entities: List[Entity], relations: Relations) -> (List[Entity], Relations):
        """Return entities with relations sorted for stable output.

        For now, this keeps entities as-is and sorts relations by (min(source,target), max(...))
        to ensure deterministic ordering when containment is present.

        Args:
            entities: Original list of entities.
            relations: Relations to be sorted into a stable order.

        Returns:
            Tuple of (entities, sorted_relations).
        """
        sorted_relations = copy.deepcopy(relations)
        sorted_relations.relations = sorted(relations.relations, key=lambda r: (min(r.source, r.target), max(r.source, r.target)))

        return entities, sorted_relations

        # sorted_entities = []
        # sorted_relations = copy.deepcopy(relations)
        # lookup_table = dict()
        # id = 0
        # # Loop over all relations, which must be in the order that "contains" relations come first.
        # for relation in relations.relations:
        #     # If the "source" (area) is not yet known, add that first
        #     if entities[relation.source] not in sorted_entities:
        #         sorted_entities.append(entities[relation.source])
        #         sorted_entities[-1].id = id
        #         lookup_table[relation.source] = id
        #         id += 1
        #     # After the "source" (area) was added, add all their "targets" (points contained within)
        #     if entities[relation.target] not in sorted_entities:
        #         sorted_entities.append(entities[relation.target])
        #         sorted_entities[-1].id = id
        #         lookup_table[relation.target] = id
        #         id += 1
        #
        # # Update the relations based on lookup table to match with the new entity IDs
        # for sorted_relation in sorted_relations.relations:
        #     sorted_relation.source = lookup_table[sorted_relation.source]
        #     sorted_relation.target = lookup_table[sorted_relation.target]
        #
        # return sorted_entities, sorted_relations

    def run(self, num_queries: int, max_number_of_entities_in_prompt: int, max_number_of_props_in_entity: int,
            prob_of_entities_with_props: float) -> List[LocPoint]:
        """Generate a batch of random query combinations.

        For each query:
          - Sample an area,
          - Generate entities (+ optional properties),
          - Generate relations,
          - Optionally sort entities/relations for containment cases.

        Args:
            num_queries: Number of query objects to generate.
            max_number_of_entities_in_prompt: Max entities allowed per query.
            max_number_of_props_in_entity: Max properties allowed per entity.
            prob_of_entities_with_props: P(entity carries ≥1 property).

        Returns:
            List of `LocPoint` objects representing the generated queries.
        """
        loc_points = []
        for _ in tqdm(range(num_queries), total=num_queries):
            area = self.generate_area()
            entities = self.generate_entities(max_number_of_entities_in_prompt=max_number_of_entities_in_prompt,
                                              max_number_of_props_in_entity=max_number_of_props_in_entity,
                                              prob_of_entities_with_props=prob_of_entities_with_props)
            relations = self.generate_relations(entities=entities)

            if relations.type in ["individual_distances_with_contains", "contains_relation"]:
                sorted_entities, sorted_relations = self.sort_entities(entities, relations)
                loc_points.append(LocPoint(area=area, entities=sorted_entities, relations=sorted_relations))
            else:
                loc_points.append(LocPoint(area=area, entities=entities, relations=relations))

        return loc_points

    def generate_area(self) -> Area:
        """Delegate to `AreaGenerator` to sample an area string/object.

        Returns:
            An `Area` describing either an empty bbox or a concrete area string.
        """
        return self.area_generator.run()


if __name__ == '__main__':
    """CLI: configure inputs & probabilities, then generate query combinations."""
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    parser.add_argument('--non_roman_vocab_file_path', help='Path to a file containing a vocabulary of areas with non-roman alphabets')
    parser.add_argument('--tag_combination_path', help='tag list file generated via retrieve_combinations')
    parser.add_argument('--tag_prop_examples_path', help='Examples of tag properties')
    parser.add_argument('--color_bundle_path', help='Path to color bundles')
    parser.add_argument('--output_file', help='File to save the output')
    parser.add_argument('--max_distance_digits', help='Define max distance', type=int)
    parser.add_argument('--write_output', action='store_true')
    parser.add_argument('--samples', help='Number of the samples to generate', type=int)
    parser.add_argument('--max_number_of_entities_in_prompt', type=int, default=4)
    parser.add_argument('--max_number_of_props_in_entity', type=int, default=4)
    parser.add_argument('--prob_of_entities_with_props', type=float, default=0.3)
    parser.add_argument('--prob_of_non_roman_areas', type=float, default=0.2)
    parser.add_argument('--prob_of_two_word_areas', type=float, default=0.5)
    parser.add_argument('--prob_adding_brand_names_as_entity', type=float, default=0.5)
    parser.add_argument('--prob_generating_contain_rel', type=float, default=0.3)
    parser.add_argument('--prob_of_rare_non_numerical_properties', type=float, default=0.1)
    parser.add_argument('--prob_of_numerical_properties', type=float, default=0.15)
    parser.add_argument('--prob_of_color_properties', type=float, default=0.15)
    parser.add_argument('--prob_of_popular_non_numerical_properties', type=float, default=0.1)
    parser.add_argument('--prob_of_other_non_numerical_properties', type=float, default=0.5)
    parser.add_argument('--prob_of_cluster_entities', type=float, default=0.3)

    args = parser.parse_args()

    tag_combination_path = args.tag_combination_path
    tag_prop_examples_path = args.tag_prop_examples_path
    geolocations_file_path = args.geolocations_file_path
    color_bundle_path = args.color_bundle_path
    non_roman_vocab_file_path = args.non_roman_vocab_file_path
    max_distance_digits = args.max_distance_digits
    num_samples = args.samples
    output_file = args.output_file
    max_number_of_entities_in_prompt = args.max_number_of_entities_in_prompt
    max_number_of_props_in_entity = args.max_number_of_props_in_entity
    prob_of_entities_with_props = args.prob_of_entities_with_props
    prob_of_two_word_areas = args.prob_of_two_word_areas
    prob_generating_contain_rel = args.prob_generating_contain_rel
    prob_of_non_roman_areas = args.prob_of_non_roman_areas
    prob_adding_brand_names_as_entity = args.prob_adding_brand_names_as_entity
    prob_of_numerical_properties = args.prob_of_numerical_properties
    prob_of_color_properties = args.prob_of_color_properties
    prob_of_other_non_numerical_properties = args.prob_of_other_non_numerical_properties
    prob_of_popular_non_numerical_properties = args.prob_of_popular_non_numerical_properties
    prob_of_rare_non_numerical_properties = args.prob_of_rare_non_numerical_properties
    prob_of_cluster_entities = args.prob_of_cluster_entities

    tag_combinations = pd.read_json(tag_combination_path, lines=True).to_dict('records')
    tag_combinations = [TagCombination(**tag_comb) for tag_comb in tag_combinations]
    property_examples = pd.read_json(tag_prop_examples_path, lines=True).to_dict('records')

    query_comb_generator = QueryCombinationGenerator(geolocation_file=geolocations_file_path,
                                                     non_roman_vocab_file=non_roman_vocab_file_path,
                                                     color_bundle_path=color_bundle_path,
                                                     tag_combinations=tag_combinations,
                                                     property_examples=property_examples,
                                                     max_distance_digits=args.max_distance_digits,
                                                     prob_of_two_word_areas=prob_of_two_word_areas,
                                                     prob_of_non_roman_areas=prob_of_non_roman_areas,
                                                     prob_generating_contain_rel=prob_generating_contain_rel,
                                                     prob_adding_brand_names_as_entity=prob_adding_brand_names_as_entity,
                                                     prob_of_numerical_properties=prob_of_numerical_properties,
                                                     prob_of_color_properties=prob_of_color_properties,
                                                     prob_of_popular_non_numerical_properties=prob_of_popular_non_numerical_properties,
                                                     prob_of_other_non_numerical_properties= prob_of_other_non_numerical_properties,
                                                     prob_of_rare_non_numerical_properties=prob_of_rare_non_numerical_properties,
                                                     prob_of_cluster_entities=prob_of_cluster_entities)

    generated_combs = query_comb_generator.run(num_queries=num_samples,
                                               max_number_of_entities_in_prompt=max_number_of_entities_in_prompt,
                                               max_number_of_props_in_entity=max_number_of_props_in_entity,
                                               prob_of_entities_with_props=prob_of_entities_with_props)

    if args.write_output:
        write_output(generated_combs, output_file=output_file)
