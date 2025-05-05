import unittest
from benchmarking.evaluate_results import AreaAnalyzer

'''Run python -m unittest benchmarking.tests.test_area_analyzer'''

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.area_analyzer = AreaAnalyzer()

    def test_compare_area(self):
        ref_area = {'type': 'bbox'}
        gen_area = {'type': 'bbox'}

        results = self.area_analyzer.compare_area(ref_area=ref_area, gen_area=gen_area)
        assert results['total_bbox'] == 1
        assert results['total_name_area'] == 0
        assert results['num_correct_bbox'] == 1
        assert results['num_correct_name_area'] == 0
        assert results['num_correct_area_type'] == 1

        ref_area = {'type': 'area', 'value': 'Olt County'}
        gen_area = {'type': 'area', 'value': 'olt county'}
        results = self.area_analyzer.compare_area(ref_area=ref_area, gen_area=gen_area)
        assert results['total_bbox'] == 0
        assert results['total_name_area'] == 1
        assert results['num_correct_bbox'] == 0
        assert results['num_correct_name_area'] == 1
        assert results['num_correct_area_type'] == 1

        ref_area = {'type': 'area', 'value': 'Munich, Bavaria'}
        gen_area = {'type': 'area', 'value': 'munich, bavaria, germany'}
        results = self.area_analyzer.compare_area(ref_area=ref_area, gen_area=gen_area)
        assert results['total_name_area'] == 1
        assert results['num_correct_area_type'] == 1
        assert results['num_correct_name_area'] == 0

        ref_area = {'type': 'area', 'value': 'São Paulo, Brazil'}
        gen_area = {'type': 'area', 'value': 'maranhão'}
        results = self.area_analyzer.compare_area(ref_area=ref_area, gen_area=gen_area)
        assert results['total_bbox'] == 0
        assert results['total_name_area'] == 1
        assert results['num_correct_bbox'] == 0
        assert results['num_correct_name_area'] == 0
        assert results['num_correct_area_type'] == 1

        ref_area = {'type': 'area', 'value': 'djerba, tunisia'}
        gen_area = {'type': 'area', 'value': 'tunesia'}
        results = self.area_analyzer.compare_area(ref_area=ref_area, gen_area=gen_area)
        assert results['total_bbox'] == 0
        assert results['total_name_area'] == 1
        assert results['num_correct_bbox'] == 0
        assert results['num_correct_name_area'] == 0
        assert results['num_correct_area_type'] == 1
