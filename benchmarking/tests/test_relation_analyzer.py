import unittest
from benchmarking.evaluate_results import RelationAnalyzer

'''Run python -m unittest benchmarking.tests.test_relation_analyzer'''

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.relation_analyzer = RelationAnalyzer()

    def test_compare_relation(self):
        ref_data = {'area': {'type': 'bbox'}, 'entities': [{'id': 0, 'type': 'nwr', 'name': 'tram stop',
                                                 'properties': [{'name': 'name', 'operator': '~', 'value': 'center'}]},
                                                {'id': 1, 'type': 'nwr', 'name': 'cafe',
                                                 'properties': [{'name': 'outdoor seating'}]},
                                                {'id': 2, 'type': 'nwr', 'name': 'brand:Burger King'}],

        'relations': [{'source': 0, 'target': 1, 'type': 'distance', 'value': '50 m', 'spatial_term': 'next to'},
                       {'source': 1, 'target': 2, 'type': 'distance', 'value': '120 m'}]}

        gen_data = {'area': {'type': 'bbox', 'value': 'bbox'}, 'entities': [{'name': 'tram stop', 'id': '0', 'type': 'nwr',
                                                                  'properties': [{'name': 'name', 'operator': '~',
                                                                                  'value': 'center'}]},
                                                                 {'name': 'cafe', 'id': '1', 'type': 'nwr',
                                                                  'properties': [{'name': 'seating', 'operator': '=',
                                                                                  'value': 'outdoor'}]},
                                                                 {'name': 'burger king', 'id': '2', 'type': 'nwr',
                                                                  'properties': []}],
         'relations': [{'source': '0', 'target': '1', 'type': 'dist', 'value': 'next to'},
                       {'source': '1', 'target': '2', 'type': 'dist', 'value': '120 meters'}]}
        results = self.relation_analyzer.compare_relations(reference_data=ref_data, generated_data=gen_data)

        self.assertEqual(results['total_rels'], 2)
        self.assertEqual(results['total_dist_rels'], 2)
        self.assertEqual(results['num_correctly_predicted_ids'], 2)
        self.assertEqual(results['num_predicted_dist_rels'], 2)
        self.assertEqual(results['num_predicted_contains_rels'], 0)
        self.assertEqual(results['num_correct_height_distance'], 1)
        self.assertEqual(results['num_correct_height_metric'], 1)
        self.assertEqual(results['total_relative_spatial_terms'],1)
        self.assertEqual(results['num_predicted_relative_spatial_terms'],0)
        self.assertEqual(results['num_predicted_relative_spatial_terms'],0)
        self.assertEqual(results['num_predicted_relative_spatial_terms'],0)