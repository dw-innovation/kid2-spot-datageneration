import unittest
from benchmarking.evaluate_results import AreaAnalyzer, PropertyAnalyzer, EntityAnalyzer, compare_yaml, ResultDataType

'''Run python -m unittest benchmarking.tests.test_entity_analyzer'''

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.area_analyzer = AreaAnalyzer()
        self.property_analyzer = PropertyAnalyzer()
        self.entity_analyzer = EntityAnalyzer(property_analyzer=self.property_analyzer)


    def test_compare_entity(self):
        entity_1 =[{'name': 'shipdock', 'id': 'entity1', 'type': 'structure', 'properties': []}, {'name': 'storage tank', 'id': 'entity2', 'type': 'infrastructure', 'properties': []}]
        entity_2 =[{'id': 0, 'type': 'nwr', 'name': 'shipdock'}, {'id': 1, 'type': 'nwr', 'name': 'storage tank'}]
        self.entity_analyzer.compare_entities(entity_1, entity_2)
