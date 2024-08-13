import unittest

import pandas as pd
from datageneration.data_model import TagProperty, Tag, TagCombination
from datageneration.generate_combination_table import QueryCombinationGenerator

'''
Execute it as follows: python -m unittest datageneration.tests.test_generate_combination_table
'''


class TestGenerateCombination(unittest.TestCase):

    def setUp(self):
        geolocations_path = 'datageneration/tests/data/countries+states+cities.json'
        tag_combinations_path = 'datageneration/tests/data/tag_combinations_v12.jsonl'
        property_examples_path = 'datageneration/tests/data/prop_examples_testv12.jsonl'

        tag_combinations = pd.read_json(tag_combinations_path, lines=True).to_dict('records')
        tag_combinations = [TagCombination(**tag_comb) for tag_comb in tag_combinations]
        property_examples = pd.read_json(property_examples_path, lines=True).to_dict('records')

        self.query_comb_generator = QueryCombinationGenerator(geolocation_file=geolocations_path,
                                                              tag_combinations=tag_combinations,
                                                              property_examples=property_examples,
                                                              max_distance_digits=5,
                                                              percentage_of_two_word_areas=0.5,
                                                              prob_generating_contain_rel=0.3,
                                                              ratio_within_radius_within=0.5)

    def test_generate_entities(self):
        entities = self.query_comb_generator.generate_entities(max_number_of_entities_in_prompt=3,
                                                               max_number_of_props_in_entity=0,
                                                               percentage_of_entities_with_props=0.3)

        assert len(entities) <= 3
        assert len(entities) > 0
        for entity in entities:
            assert len(entity.properties) == 0

        entities = self.query_comb_generator.generate_entities(max_number_of_entities_in_prompt=3,
                                                               max_number_of_props_in_entity=4,
                                                               percentage_of_entities_with_props=0.3)

        assert len(entities) <= 3
        assert len(entities) > 0
        for entity in entities:
            assert len(entity.properties) <= 4

    def test_property_generate(self):
        candidate_properties = [
            TagProperty(descriptors=["name"], tags=[Tag(key="name", operator="~", value="***example***")]),
            TagProperty(descriptors=['street name'], tags=[Tag(key="addr:street", operator="~", value="***example***")]),
            TagProperty(descriptors=['house number', 'building number'], tags=[Tag(key="addr:housenumber", operator="~", value="***example***")]),
            TagProperty(descriptors=["height"], tags=[Tag(key="height", operator="=", value="***numeric***")])
        ]
        properties = self.query_comb_generator.generate_properties(
            candidate_properties=candidate_properties,
            num_of_props=4)

        assert len(properties) == 4

        properties = self.query_comb_generator.generate_properties(candidate_properties=candidate_properties,
                                                                   num_of_props=3)

        assert len(properties) == 3


if __name__ == '__main__':
    unittest.main()
