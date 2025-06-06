import unittest
from benchmarking.evaluate_results import RelationAnalyzer

'''Run python -m unittest benchmarking.tests.test_relation_analyzer'''

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.relation_analyzer = RelationAnalyzer()

    def test_compare_relation(self):
        ref_data = {'area': {'type': 'bbox'}, 'entities': [{'id': 0, 'type': 'cluster', 'name': 'tall building', 'minPoints': 2, 'maxDistance': '300 m', 'normalized_name': 'high-rise building'}, {'id': 1, 'type': 'nwr', 'name': 'river', 'normalized_name': 'canal'}, {'id': 2, 'type': 'nwr', 'name': 'house', 'properties': [{'name': 'color', 'operator': '=', 'value': 'yellow', 'normalized_name': 'building color'}], 'normalized_name': 'linked house'}], 'relations': [{'type': 'distance', 'source': 0, 'target': 1, 'value': '50 m', 'spatial_term': 'near'}, {'type': 'contains', 'source': 1, 'target': 2}]}
        gen_data = {'area': {'type': 'bbox'}, 'entities': [{'id': 0, 'name': 'tall building', 'type': 'nwr', 'normalized_name': 'high-rise building'}, {'id': 1, 'name': 'river', 'properties': [{'name': 'house colour', 'operator': '=', 'value': 'yellow'}], 'type': 'nwr', 'normalized_name': 'canal'}], 'relations': [{'source': 0, 'target': 1, 'type': 'distance', 'value': '300 m'}]}
        full_paired_ent = [({'id': 0, 'type': 'cluster', 'name': 'tall building', 'minPoints': 2, 'maxDistance': '300 m', 'normalized_name': 'high-rise building'}, {'id': 0, 'name': 'tall building', 'type': 'nwr', 'normalized_name': 'high-rise building'}), ({'id': 1, 'type': 'nwr', 'name': 'river', 'normalized_name': 'canal'}, {'id': 1, 'name': 'river', 'properties': [{'name': 'house colour', 'operator': '=', 'value': 'yellow'}], 'type': 'nwr', 'normalized_name': 'canal'}), ({'id': 2, 'type': 'nwr', 'name': 'house', 'properties': [{'name': 'color', 'operator': '=', 'value': 'yellow', 'normalized_name': 'building color'}], 'normalized_name': 'linked house'}, {'id': 0, 'name': 'tall building', 'type': 'nwr', 'normalized_name': 'high-rise building'})]
        results = self.relation_analyzer.compare_relations(reference_data=ref_data, generated_data=gen_data, full_paired_entities=full_paired_ent)
        self.assertEqual(results['total_dist_rels'], 1)
        self.assertEqual(results['total_contains_rels'], 1)
        self.assertEqual(results['num_correct_dist_rels'], 0)
        self.assertEqual(results['num_correct_contains_rels'], 0)
        self.assertEqual(results['num_correct_dist_edges'], 1)
        self.assertEqual(results['num_correct_dist_value'], 0)
        self.assertEqual(results['num_correct_dist_metric'], 1)
        self.assertEqual(results['num_correct_dist'], 0)
        self.assertEqual(results['num_correct_rel_type'], 1)
        self.assertEqual(results['total_relative_spatial_terms'],1)
        self.assertEqual(results['num_correct_relative_spatial_terms'],0)

    def test_compare_relation(self):
        ref_data = {'area': {'type': 'area', 'value': 'ស្មរ៉ុមណេដ្រេ តំបន់រាជធានីនៃប្រទេសដាណឺម៉ាក'}, 'entities': [
            {'id': 0, 'type': 'nwr', 'name': 'clothing store',
             'properties': [{'name': 'name', 'operator': '~', 'value': 'h&m', 'normalized_name': 'brand'}],
             'normalized_name': 'fashion outlet'}]}
        gen_data = {'area': {'type': 'area', 'value': 'ស្មរ៉ុមណេដ្រេ, តំបន់រាជធានីនៃប្រទេសដាណឺម៉ាក, ដាណឺម៉ាក'}, 'entities': [
            {'id': 0, 'name': 'clothing store',
             'properties': [{'name': 'name', 'operator': '~', 'value': 'h&m', 'normalized_name': 'brand'}],
             'type': 'nwr', 'normalized_name': 'fashion outlet'}]}
        full_paired_ent =[({'id': 0, 'type': 'nwr', 'name': 'clothing store',
           'properties': [{'name': 'name', 'operator': '~', 'value': 'h&m', 'normalized_name': 'brand'}],
           'normalized_name': 'fashion outlet'}, {'id': 0, 'name': 'clothing store', 'properties': [
            {'name': 'name', 'operator': '~', 'value': 'h&m', 'normalized_name': 'brand'}], 'type': 'nwr',
                                                  'normalized_name': 'fashion outlet'})]

        results = self.relation_analyzer.compare_relations(reference_data=ref_data, generated_data=gen_data, full_paired_entities=full_paired_ent)
        self.assertTrue(results['relation_perfect_result'])

        ref_data = {'area': {'type': 'area', 'value': 'Bay Harbor'},
         'entities': [{'id': 0, 'type': 'nwr', 'name': 'artwork', 'normalized_name': 'bust'},
                      {'id': 1, 'type': 'nwr', 'name': 'music venue', 'normalized_name': 'concert hallstage'},
                      {'id': 2, 'type': 'nwr', 'name': 'library', 'normalized_name': 'media center'}],
         'relations': [{'type': 'distance', 'source': 0, 'target': 1, 'value': '100 m'},
                       {'type': 'distance', 'source': 1, 'target': 2, 'value': '0.5 mi'}]}
        gen_data = {'area': {'type': 'area', 'value': 'bay harbor'}, 'entities': [
            {'id': 0, 'maxdistance': '100 m', 'minpoints': 3, 'name': 'music venue', 'type': 'cluster',
             'normalized_name': 'concert hallstage'},
            {'id': 1, 'name': 'artwork', 'type': 'nwr', 'normalized_name': 'bust'},
            {'id': 2, 'name': 'library', 'type': 'nwr', 'normalized_name': 'media center'}],
         'relations': [{'source': 0, 'target': 1, 'type': 'distance', 'value': '2000 m'},
                       {'source': 0, 'target': 2, 'type': 'distance', 'value': '0.5 mi'}]}
        full_paired_ent = [({'id': 0, 'type': 'nwr', 'name': 'artwork', 'normalized_name': 'bust'},
          {'id': 1, 'name': 'artwork', 'type': 'nwr', 'normalized_name': 'bust'}),
         ({'id': 1, 'type': 'nwr', 'name': 'music venue', 'normalized_name': 'concert hallstage'},
          {'id': 0, 'maxdistance': '100 m', 'minpoints': 3, 'name': 'music venue', 'type': 'cluster',
           'normalized_name': 'concert hallstage'}),
         ({'id': 2, 'type': 'nwr', 'name': 'library', 'normalized_name': 'media center'},
          {'id': 2, 'name': 'library', 'type': 'nwr', 'normalized_name': 'media center'})]

        results = self.relation_analyzer.compare_relations(reference_data=ref_data, generated_data=gen_data, full_paired_entities=full_paired_ent)
        self.assertFalse(results['relation_perfect_result'])