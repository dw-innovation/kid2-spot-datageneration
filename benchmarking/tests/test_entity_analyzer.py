import unittest
from benchmarking.evaluate_results import AreaAnalyzer, EntityAndPropertyAnalyzer, compare_yaml, ResultDataType

'''Run python -m unittest benchmarking.tests.test_entity_analyzer'''

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.area_analyzer = AreaAnalyzer()
        self.entity_analyzer = EntityAndPropertyAnalyzer()
    def test_compare_entity(self):
        entity_1 =[{'name': 'shipdock', 'id': 'entity1', 'type': 'structure'}, {'name': 'storage tank', 'id': 'entity2', 'type': 'infrastructure'}]
        entity_2 =[{'id': 0, 'type': 'nwr', 'name': 'shipdock'}, {'id': 1, 'type': 'nwr', 'name': 'storage tank'}]
        self.entity_analyzer.compare_entities(entity_1, entity_2)


    def test_compare_cluster_entity(self):
        entity_1 =[{'id': 0, 'type': 'cluster', 'name': 'mosques', 'minPoints': 3, 'maxDistance': '150 m'}, {'id': 1, 'name': 'pharmacy', 'type': 'nwr'}]
        entity_2 =[{'id': 0, 'maxdistance': '150 in', 'minpoints': 3, 'name': 'mosque', 'type': 'cluster'}, {'id': 1, 'maxdistance': '300 m', 'minpoints': 1, 'name': 'pharmacy', 'type': 'cluster'}]
        results = self.entity_analyzer.compare_entities(entity_1, entity_2)
        self.assertTrue(results['num_correct_entity_type'], 1)

        entity_1 = [{'id': 0, 'type': 'cluster', 'name': 'payed parking spot', 'minPoints': 3, 'maxDistance': '50 m'},
         {'id': 1, 'name': 'church', 'type': 'nwr'}, {'id': 2, 'name': 'tower', 'type': 'cluster', 'minPoints':2, 'maxDistance': '50 mi', 'properties': [
            {'name': 'height', 'operator': '=', 'value': '100 m'}]}]
        entity_2 = [{'id': 0, 'maxdistance': '2000 m', 'minpoints': 3, 'name': 'paid parking', 'type': 'cluster'},
         {'id': 1, 'maxdistance': '100 m', 'minpoints': 1, 'name': 'church', 'type': 'cluster'},
         {'id': 2, 'maxdistance': '100 m', 'minpoints': 2, 'name': 'tower',
          'properties': [{'name': 'height', 'operator': '=', 'value': '100 mi'}], 'type': 'cluster'}]

        results = self.entity_analyzer.compare_entities(reference_entities=entity_1, predicted_entities=entity_2)
        self.assertEqual(results['num_correct_entity_type'], 2)
        self.assertEqual(results['num_correct_cluster_points'], 2)
        self.assertEqual(results['num_correct_height_distance'], 1)
        self.assertEqual(results['num_correct_height_metric'], 0)

        print(results)

        _, paired_entities, unpaired_entities = self.entity_analyzer.pair_objects(entity_1, entity_2)
        paired_entities = [*sum(paired_entities, ())]
        for pair_entity in paired_entities:
            assert pair_entity in ['paid parking', 'payed parking spot', 'church', 'tower']

        assert len(unpaired_entities['reference']) == 0
        assert len(unpaired_entities['prediction']) == 0

    def test_properties(self):
        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'car shop', 'properties': [{'name': 'floors', 'operator': '=', 'value': 2}, {'name': 'name', 'operator': '~', 'value': 'Master Farma'}]}, {'id': 1, 'type': 'nwr', 'name': 'bus stop'}, {'id': 2, 'type': 'nwr', 'name': 'aqueduct'}]
        entity_2 = [{'id': 0, 'name': 'car shop', 'properties': [{'name': 'name', 'operator': '~', 'value': 'master farma'}, {'name': 'floors', 'operator': '=', 'value': '2'}], 'type': 'nwr'}, {'id': 1, 'name': 'bus stop', 'type': 'nwr'}, {'id': 2, 'name': 'aqueduct', 'type': 'nwr'}]

        results = self.entity_analyzer.compare_entities(entity_1, entity_2)

        assert results['total_properties'] >= results['num_correct_properties_weak']