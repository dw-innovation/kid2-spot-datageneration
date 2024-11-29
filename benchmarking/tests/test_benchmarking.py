import unittest
from benchmarking.evaluate_results import AreaAnalyzer, PropertyAnalyzer, EntityAnalyzer, compare_yaml, ResultDataType

'''Run python -m unittest benchmarking.tests.test_benchmarking'''

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.area_analyzer = AreaAnalyzer()
        self.property_analyzer = PropertyAnalyzer()
        self.entity_analyzer = EntityAnalyzer(property_analyzer=self.property_analyzer)


    def test_compare_yaml(self):
        yaml_pred_string = '''area:
  type: area
  value: ស្មរ៉ុមណេដ្រេ តំបន់រាជធានីនៃប្រទេសដាណឺម៉ាក
entities:
  - id: 0
    name: clothing store
    properties:
     - name: brand
       operator: '~'
       value: h&m
    type: nwr'''
        yaml_true_string = '''area:
  type: area
  value: ស្មរ៉ុមណេដ្រេ តំបន់រាជធានីនៃប្រទេសដាណឺម៉ាក
entities:
  - id: 0
    type: nwr
    name: clothing store
    properties:
      - name: brand name
        operator: '~'
        value: H&M'''
        comparision_result = compare_yaml(area_analyzer=self.area_analyzer,
                                          entity_analyzer=self.entity_analyzer,
                                          property_analyzer=self.property_analyzer,
                                          yaml_true_string=yaml_true_string,
                                          yaml_pred_string=yaml_pred_string)
        self.assertEquals(ResultDataType.TRUE, comparision_result.are_properties_same)



        yaml_true_string='''area:
  type: area
  value: new delhi, india 
entities:
  - id: 0
    type: nwr
    name: caffee place
    properties:
      - name: name
        operator: '~'
        value: 'in dia'
      - name: outdoor seating
  - id: 1
    type: nwr
    name: pharmacy
  - id: 2
    type: nwr
    name: historic monument
relations:
  - source: 0
    target: 1
    type: distance
    value: 50 m
  - source: 0
    target: 2
    type: distance
    value: 50 m
'''
        yaml_pred_string = '''area:
      type: area
      value: new delhi, india
    entities:
    - id: 0
      name: coffee place
      properties:
      - name: name
        operator: '~'
        value: in dia
      - name: outdoor seating
      type: nwr
    - id: 1
      name: pharmacy
      type: nwr
    - id: 2
      name: historic monument
      type: nwr
    relations:
    - source: 0
      target: 1
      type: distance
      value: 50 m
    - source: 0
      target: 2
      type: distance
      value: 50 m'''
        comparision_result = compare_yaml(area_analyzer=self.area_analyzer,
                                          entity_analyzer=self.entity_analyzer,
                                          property_analyzer=self.property_analyzer,
                                          yaml_true_string=yaml_true_string,
                                          yaml_pred_string=yaml_pred_string)
        self.assertEquals(ResultDataType.TRUE, comparision_result.are_properties_same)