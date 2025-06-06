import unittest
from benchmarking.evaluate_results import AreaAnalyzer, EntityAndPropertyAnalyzer, compare_yaml
from benchmarking.utils import load_key_table

'''Run python -m unittest benchmarking.tests.test_entity_analyzer'''

class TestBenchmarking(unittest.TestCase):
    def setUp(self):
        self.area_analyzer = AreaAnalyzer()
        descriptors = load_key_table(path='datageneration/data/Spot_primary_keys_bundles.xlsx')
        self.entity_analyzer = EntityAndPropertyAnalyzer(descriptors=descriptors)

    def test_perfect_match(self):
        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'car shop', 'properties': [{'name': 'floors', 'operator': '=', 'value': 2}, {'name': 'name', 'operator': '~', 'value': 'Master Farma'}]}, {'id': 1, 'type': 'nwr', 'name': 'bus stop'}, {'id': 2, 'type': 'nwr', 'name': 'aqueduct'}]
        entity_2 = [{'id': 0, 'name': 'bus stop', 'type': 'nwr'}, {'id': 1, 'name': 'car shop', 'properties': [{'name': 'etage', 'operator': '=', 'value': '2'}, {'name': 'name', 'operator': '~', 'value': 'master farma'}], 'type': 'nwr'}, {'id': 2, 'name': 'aqueduct', 'type': 'nwr'}]
        results, _ = self.entity_analyzer.compare_entities(entity_1, entity_2)
        self.assertTrue(results['entity_perfect_result'])
        self.assertTrue(results['props_perfect_result'])

        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'clothing store', 'properties': [{'name': 'name', 'operator': '~', 'value': 'H&M'}]}]
        entity_2 = [{'id': 0, 'name': 'clothing store', 'properties': [{'name': 'name', 'operator': '~', 'value': 'h&m'}], 'type': 'nwr'}]

        results, _ = self.entity_analyzer.compare_entities(entity_1, entity_2)
        self.assertTrue(results['entity_perfect_result'])
        self.assertTrue(results['props_perfect_result'])

        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'artwork'}, {'id': 1, 'type': 'nwr', 'name': 'music venue'}, {'id': 2, 'type': 'nwr', 'name': 'library'}]
        entity_2 = [{'id': 0, 'maxdistance': '100 m', 'minpoints': 3, 'name': 'music venue', 'type': 'cluster'}, {'id': 1, 'name': 'artwork', 'type': 'nwr'}, {'id': 2, 'name': 'library', 'type': 'nwr'}]
        results, _ = self.entity_analyzer.compare_entities(entity_1, entity_2)

        self.assertFalse(results['entity_perfect_result'])
        self.assertTrue(results['props_perfect_result'])

        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'social facility',
          'properties': [{'name': 'levels', 'operator': '<', 'value': 3},
                         {'name': 'name', 'operator': '~', 'value': 'ole'}]},
         {'id': 1, 'type': 'nwr', 'name': 'petrol station'}, {'id': 2, 'type': 'nwr', 'name': 'fabric shop'}]
        entity_2 = [{'id': 0, 'name': 'social facility', 'properties': [{'name': 'levels', 'operator': '<', 'value': '3'},
                                                             {'name': 'name', 'operator': '~', 'value': 'ole'}],
                     'type': 'nwr'}, {'id': 1, 'name': 'petroleum station', 'type': 'nwr'},
         {'id': 2, 'name': 'fabric shop', 'type': 'nwr'}]

        results, _ = self.entity_analyzer.compare_entities(entity_1, entity_2)
        self.assertTrue(results['entity_perfect_result'])


    def test_compare_entity(self):
        entity_1 =[{'name': 'shipdock', 'id': 'entity1', 'type': 'structure'}, {'name': 'storage tank', 'id': 'entity2', 'type': 'infrastructure'}]
        entity_2 =[{'id': 0, 'type': 'nwr', 'name': 'shipdock'}, {'id': 1, 'type': 'nwr', 'name': 'storage tank'}]
        self.entity_analyzer.compare_entities(entity_1, entity_2)

        entity_1 = [{'id': 0, 'name': 'park', 'type': 'nwr', 'normalized_name': 'public garden'}, {'id': 1, 'name': 'soccer field', 'type': 'nwr', 'normalized_name': 'soccer facility'}, {'id': 2, 'name': 'hut', 'properties': [{'name': 'roof colour', 'operator': '=', 'value': 'red', 'normalized_name': 'building color'}], 'type': 'nwr', 'normalized_name': 'shack'}]
        entity_2 = [{'id': 0, 'name': 'soccer field', 'type': 'nwr', 'normalized_name': 'soccer facility'}, {'id': 1, 'name': 'hut', 'properties': [{'name': 'roof material', 'operator': '=', 'value': 'roof-tiles', 'normalized_name': 'roofing material'}, {'name': 'roof colour', 'operator': '=', 'value': 'red', 'normalized_name': 'building color'}], 'type': 'nwr', 'normalized_name': 'shack'}, {'id': 2, 'name': 'park', 'type': 'nwr', 'normalized_name': 'public garden'}]
        results, _ =  self.entity_analyzer.compare_entities(entity_1, entity_2)

        self.assertTrue(results['total_color_property']==1)
        self.assertTrue(results['num_correct_color']==1)

        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'military base'}, {'id': 1, 'type': 'nwr', 'name': 'airport', 'normalized_name': 'air base'}, {'id': 2, 'type': 'cluster', 'name': 'building', 'minPoints': 3, 'maxDistance': '50 m', 'properties': [{'name': 'roof color', 'operator': '=', 'value': 'red'}]}]
        entity_2 = [{'id': 0, 'name': 'military base', 'type': 'nwr'}, {'id': 1, 'name': 'airport', 'properties': [{'name': 'roof material', 'operator': '=', 'value': 'roof_red'}, {'name': 'roof material', 'operator': '=', 'value': 'roof_red'}, {'name': 'roof material', 'operator': '=', 'value': 'roof_red'}], 'type': 'nwr'}]

        results, _ = self.entity_analyzer.compare_entities(reference_entities=entity_1, predicted_entities=entity_2)
        self.assertTrue(results['num_entity_match_weak']==2)
        self.assertTrue(results['num_missing_properties']==1)
        self.assertTrue(results['num_hallucinated_properties']==3)

        entity_1 = [{'id': 0, 'type': 'cluster', 'name': 'salt bin', 'minPoints': 8, 'maxDistance': '150 m'}]
        entity_2 = [{'id': 0, 'name': 'salt bin', 'type': 'nwr'}, {'id': 1, 'name': 'salt bin', 'type': 'nwr'}, {'id': 2, 'name': 'salt bin', 'type': 'nwr'}, {'id': 3, 'name': 'salt bin', 'type': 'nwr'}, {'id': 4, 'name': 'salt bin', 'type': 'nwr'}, {'id': 5, 'name': 'salt bin', 'type': 'nwr'}, {'id': 6, 'name': 'salt bin', 'type': 'nwr'}, {'id': 7, 'name': 'salt bin', 'type': 'nwr'}]
        results, _ = self.entity_analyzer.compare_entities(reference_entities=entity_1, predicted_entities=entity_2)
        self.assertTrue(results['num_hallucinated_entity']==7)

    def test_pair_objects(self):
        obj_1 = [{'id': 0, 'name': 'social facility', 'properties': [{'name': 'levels', 'operator': '<', 'value': '3'}, {'name': 'name', 'operator': '~', 'value': 'ole'}], 'type': 'nwr'}, {'id': 1, 'name': 'petroleum station', 'type': 'nwr'}, {'id': 2, 'name': 'fabric shop', 'type': 'nwr'}]
        obj_2 = [{'id': 0, 'type': 'nwr', 'name': 'social facility', 'properties': [{'name': 'levels', 'operator': '<', 'value': 3}, {'name': 'name', 'operator': '~', 'value': 'ole'}]}, {'id': 1, 'type': 'nwr', 'name': 'petrol station'}, {'id': 2, 'type': 'nwr', 'name': 'fabric shop'}]

        full_paired_entities, paired_objs, full_unpaired_objs, unpaired_objs = self.entity_analyzer.pair_objects(obj_1, obj_2)
        assert len(paired_objs) == 3
        assert len(unpaired_objs['reference']) == 0

        obj_1 = [{'name': 'etage', 'operator': '=', 'value': '2'}, {'name': 'name', 'operator': '~', 'value': 'master farma'}]
        obj_2 = [{'name': 'floors', 'operator': '=', 'value': '2'}, {'name': 'name', 'operator': '~', 'value': 'master farma'}]

        full_paired_entities, paired_objs, full_unpaired_objs, unpaired_objs = self.entity_analyzer.pair_objects(obj_1, obj_2)
        assert len(unpaired_objs['reference']) == 0

        obj_1 = [{'id': 0, 'type': 'cluster', 'name': 'dm market', 'minPoints': 3, 'maxDistance': '2000 m'}, {'id': 1, 'name': 'burger king', 'type': 'nwr'}]
        obj_2 = [{'id': 0, 'name': 'brand:dm', 'type': 'nwr'}, {'id': 1, 'name': 'brand:burger king', 'type': 'nwr'}]

        full_paired_entities, paired_objs, full_unpaired_objs, unpaired_objs = self.entity_analyzer.pair_objects(obj_1, obj_2)
        print(full_paired_entities)
        assert len(unpaired_objs['reference']) == 0

        obj_1 = [{'id': 0, 'type': 'nwr', 'name': 'military base'},
                    {'id': 1, 'type': 'nwr', 'name': 'airport', 'normalized_name': 'air base'},
                    {'id': 2, 'type': 'cluster', 'name': 'building', 'minPoints': 3, 'maxDistance': '50 m',
                     'properties': [{'name': 'roof color', 'operator': '=', 'value': 'red'}]}]
        obj_2 = [{'id': 0, 'name': 'military base', 'type': 'nwr'}, {'id': 1, 'name': 'airport', 'properties': [
            {'name': 'roof material', 'operator': '=', 'value': 'roof_red'},
            {'name': 'roof material', 'operator': '=', 'value': 'roof_red'},
            {'name': 'roof material', 'operator': '=', 'value': 'roof_red'}], 'type': 'nwr'}]


        full_paired_entities, paired_objs, full_unpaired_objs, unpaired_objs = self.entity_analyzer.pair_objects(reference_objs=obj_1, predicted_objs=obj_2)
        self.assertTrue(len(paired_objs) == 2)
        self.assertTrue(len(full_paired_entities) == 2)
        self.assertTrue(len(unpaired_objs['reference']) == 1)

        obj_1 = [{'id': 0, 'type': 'cluster', 'name': 'salt bin', 'minPoints': 8, 'maxDistance': '150 m'}]
        obj_2 = [{'id': 0, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}, {'id': 1, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}, {'id': 2, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}, {'id': 3, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}, {'id': 4, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}, {'id': 5, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}, {'id': 6, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}, {'id': 7, 'name': 'salt bin', 'type': 'nwr', 'normalized_name': 'grit bin'}]
        full_paired_entities, paired_objs, full_unpaired_objs, unpaired_objs = self.entity_analyzer.pair_objects(reference_objs=obj_1,
                                                                                             predicted_objs=obj_2)
        self.assertTrue(len(unpaired_objs['prediction']), 7)

    def test_compare_cluster_entity(self):
        entity_1 =[{'id': 0, 'type': 'cluster', 'name': 'mosques', 'minPoints': 3, 'maxDistance': '150 m'}, {'id': 1, 'name': 'pharmacy', 'type': 'nwr'}]
        entity_2 =[{'id': 0, 'maxdistance': '150 in', 'minpoints': 3, 'name': 'mosque', 'type': 'cluster'}, {'id': 1, 'maxdistance': '300 m', 'minpoints': 1, 'name': 'pharmacy', 'type': 'cluster'}]
        (results, _) = self.entity_analyzer.compare_entities(entity_1, entity_2)
        print('==results==')
        print(results)
        self.assertTrue(results['num_correct_entity_type']== 1)

        entity_1 = [{'id': 0, 'type': 'cluster', 'name': 'payed parking spot', 'minPoints': 3, 'maxDistance': '50 m'},
         {'id': 1, 'name': 'church', 'type': 'nwr'}, {'id': 2, 'name': 'tower', 'type': 'cluster', 'minPoints':2, 'maxDistance': '50 mi', 'properties': [
            {'name': 'height', 'operator': '=', 'value': '100 m'}]}]
        entity_2 = [{'id': 0, 'maxdistance': '2000 m', 'minpoints': 3, 'name': 'paid parking', 'type': 'cluster'},
         {'id': 1, 'maxdistance': '100 m', 'minpoints': 1, 'name': 'church', 'type': 'cluster'},
         {'id': 2, 'maxdistance': '100 m', 'minpoints': 2, 'name': 'tower',
          'properties': [{'name': 'height', 'operator': '=', 'value': '100 mi'}], 'type': 'cluster'}]

        (results, _) = self.entity_analyzer.compare_entities(reference_entities=entity_1, predicted_entities=entity_2)
        self.assertEqual(results['num_correct_entity_type'],2)
        self.assertEqual(results['num_correct_cluster_points'],2)
        self.assertEqual(results['num_correct_height_distance'],1)
        self.assertEqual(results['num_correct_height_metric'], 0)

        _, paired_entities, _, unpaired_entities = self.entity_analyzer.pair_objects(entity_1, entity_2)
        paired_entities = [*sum(paired_entities, ())]
        assert len(paired_entities) == 6
        assert len(unpaired_entities['reference']) == 0
        assert len(unpaired_entities['prediction']) == 0


        entity_1 = [{'id': 0, 'maxdistance': '150 m', 'minpoints': 8, 'name': 'salt bin', 'type': 'cluster'}]
        entity_2 = [{'id': 0, 'type': 'cluster', 'name': 'salt bin', 'minPoints': 8, 'maxDistance': '150 m'}]
        (results, _) = self.entity_analyzer.compare_entities(reference_entities=entity_1, predicted_entities=entity_2)
        self.assertEqual(results['entity_perfect_result'], 1)

    def test_pairing(self):
        ref_properties = [{'name': 'building material', 'operator': '=', 'value': 'brick'}, {'name': 'height', 'operator': '<', 'value': '2 m'}]
        gen_properties = [{'name': 'height', 'operator': '<', 'value': '2 m'}, {'name': 'material', 'operator': '=', 'value': 'brick'}]
        full_paired_props, paired_props, full_unpaired_props, unpaired_props = self.entity_analyzer.pair_objects(predicted_objs=gen_properties, reference_objs=ref_properties)

        assert len(full_paired_props) == 2
        assert len(paired_props) == 2
        assert len(unpaired_props['reference']) == 0
        assert len(unpaired_props['prediction']) == 0

        ref_properties = [{'name': 'roof colour', 'operator': '=', 'value': 'red'}]
        gen_properties = [{'name': 'roof material', 'operator': '=', 'value': 'roof-tiles'}, {'name': 'hue', 'operator': '=', 'value': 'red'}]
        full_paired_props, paired_props, full_unpaired_props, unpaired_props = self.entity_analyzer.pair_objects(
            predicted_objs=gen_properties, reference_objs=ref_properties)

        assert len(full_paired_props) == 1
        assert len(paired_props) == 1
        assert len(unpaired_props['prediction']) == 1

        ref_entities = [{'id': 0, 'type': 'nwr', 'name': 'park'}, {'id': 1, 'type': 'nwr', 'name': 'football field'}, {'id': 2, 'type': 'nwr', 'name': 'fountain'}]
        gen_entities = [{'id': 0, 'name': 'football field', 'type': 'nwr'}, {'id': 1, 'name': 'fountaine', 'type': 'nwr'}, {'id': 2, 'name': 'park', 'type': 'nwr'}]
        full_paired_entities, paired_ents, full_unpaired_ents, unpaired_ents = self.entity_analyzer.pair_objects(
            predicted_objs=gen_entities, reference_objs=ref_entities)
        assert len(unpaired_ents['prediction']) == 0


        ref_entities = [{'id': 0, 'name': 'church', 'type': 'nwr', 'properties': [{'name': 'religion', 'operator': '=', 'value': 'christian'}]}, {'id': 1, 'name': 'fire tower', 'type': 'nwr'}, {'id': 2, 'name': 'houseboat', 'type': 'nwr'}]
        gen_entities = [{'id': 0, 'name': 'christian church', 'type': 'nwr'}, {'id': 1, 'name': 'fire tower', 'type': 'nwr'}, {'id': 2, 'name': 'houseboat', 'type': 'nwr'}]
        full_paired_entities, paired_ents, full_unpaired_ents, unpaired_ents  = self.entity_analyzer.pair_objects(
            predicted_objs=gen_entities, reference_objs=ref_entities)
        print('unpaired ents!!!')
        print(full_unpaired_ents)
        self.assertEqual(len(unpaired_ents['prediction']), 0)

        ref_props =[{'name': 'cuisine', 'operator': '=', 'value': 'italian'}, {'name': 'outdoor seating'}]
        ent_props = [{'name': 'outdoor seating'}]

        full_paired_props, paired_props, full_unpaired_props, unpaired_props  = self.entity_analyzer.pair_objects(predicted_objs=ref_props, reference_objs=ent_props)
        self.assertEqual(len(paired_props), 1)
        self.assertEqual(len(unpaired_props['prediction']), 1)
        self.assertEqual(len(full_paired_props), 1)


        ref_props = [{'name': 'name', 'operator': '~', 'value': 'မက်ဒေါ်နယ်'}, {'name': 'name', 'operator': '~', 'value': 'ဘာဂါကင်း'}]
        ent_props = [{'name': 'name', 'operator': '~', 'value': 'ဘာဂါကင်း'}]
        full_paired_props, paired_props, full_unpaired_props, unpaired_props = self.entity_analyzer.pair_objects(
            reference_objs=ref_props, predicted_objs=ent_props)

        self.assertEqual(len(full_paired_props), 1)
        self.assertEqual(len(unpaired_props['reference']), 1)

        ref_entities = [{'id': 0, 'name': 'restaurant', 'type': 'nwr', 'properties': [{'name': 'name', 'operator': '~', 'value': 'မက်ဒေါ်နယ်'}]}, {'id': 1, 'name': 'restaurant', 'type': 'nwr', 'properties': [{'name': 'name', 'operator': '~', 'value': 'ဘာဂါကင်း'}]}]
        pred_entities = [{'id': 0, 'name': 'restaurant', 'properties': [{'name': 'name', 'operator': '~', 'value': 'မက်ဒေါ်နယ်'}, {'name': 'name', 'operator': '~', 'value': 'ဘာဂါကင်း'}], 'type': 'nwr'}]
        results = self.entity_analyzer.pair_objects(reference_objs=ref_entities, predicted_objs=pred_entities)

        ref_entities = [{'id': 0, 'type': 'cluster', 'name': 'atm', 'minPoints': 8, 'maxDistance': '2 km', 'properties': [{'name': 'name', 'operator': '~', 'value': 'santander'}]}]
        pred_entities = [{'id': 0, 'name': "brand:eight santander atm's", 'type': 'nwr'}, {'id': 1, 'name': "brand:eight santander atm's", 'type': 'nwr'}]
        full_paired_props, paired_props, full_unpaired_props, unpaired_props = self.entity_analyzer.pair_objects(reference_objs=ref_entities, predicted_objs=pred_entities)
        self.assertEqual(len(unpaired_props['prediction']), 2)



    def test_properties(self):
        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'car shop', 'properties': [{'name': 'floors', 'operator': '=', 'value': 2}, {'name': 'name', 'operator': '~', 'value': 'Master Farma'}]}, {'id': 1, 'type': 'nwr', 'name': 'bus stop'}, {'id': 2, 'type': 'nwr', 'name': 'aqueduct'}]
        entity_2 = [{'id': 0, 'name': 'car shop', 'properties': [{'name': 'name', 'operator': '~', 'value': 'master farma'}, {'name': 'floors', 'operator': '=', 'value': '2'}], 'type': 'nwr'}, {'id': 1, 'name': 'bus stop', 'type': 'nwr'}, {'id': 2, 'name': 'aqueduct', 'type': 'nwr'}]

        (results, _) = self.entity_analyzer.compare_entities(entity_1, entity_2)

        entity_1 = [{'id': 0, 'type': 'nwr', 'name': 'pergola',
                     'properties': [{'name': 'building material', 'operator': '=', 'value': 'brick'},
                                    {'name': 'height', 'operator': '<', 'value': '2 m'}]},
                    {'id': 1, 'type': 'nwr', 'name': 'high-rise apartment',
                     'properties': [{'name': 'house number', 'operator': '=', 'value': '384-A'}]},
                    {'id': 2, 'type': 'nwr', 'name': 'charging base'}]

        entity_2 = [{'id': 0, 'name': 'pergola', 'properties': [{'name': 'height', 'operator': '<', 'value': '2 m'},
                                                                {'name': 'material', 'operator': '=',
                                                                 'value': 'brick'}], 'type': 'nwr'},
                    {'id': 1, 'name': 'high-rise apartment',
                     'properties': [{'name': 'house number', 'operator': '=', 'value': '384-a'}], 'type': 'nwr'},
                    {'id': 2, 'name': 'charging base', 'type': 'nwr'}]

        (results, _) = self.entity_analyzer.compare_entities(entity_1, entity_2)
        self.assertEqual(results['num_hallucinated_properties'], 0)
        self.assertEqual(results['num_correct_properties_perfect'], 3)
        self.assertEqual(results['num_correct_height'], 1)
        self.assertEqual(results['num_correct_height_metric'], 1)
        self.assertEqual(results['num_correct_height_distance'], 1)

        entity_1 = [{'id': 0, 'name': 'park', 'type': 'nwr'}, {'id': 1, 'name': 'soccer field', 'type': 'nwr'}, {'id': 2, 'name': 'hut', 'properties': [{'name': 'roof colour', 'operator': '=', 'value': 'red'}], 'type': 'nwr'}]
        entity_2 = [{'id': 0, 'name': 'soccer field', 'type': 'nwr'}, {'id': 1, 'name': 'hut', 'properties': [{'name': 'roof material', 'operator': '=', 'value': 'roof-tiles'}, {'name': 'hue', 'operator': '=', 'value': 'red'}], 'type': 'nwr'}, {'id': 2, 'name': 'park', 'type': 'nwr'}]

        (results, _) = self.entity_analyzer.compare_entities(entity_1, entity_2)
        self.assertEqual(results['total_color_property'], 1)
        self.assertEqual(results['num_hallucinated_properties'], 1)
        self.assertEqual(results['num_missing_properties'],0)
        self.assertEqual(results['num_correct_properties_perfect'], 1)
        self.assertEqual(results['num_correct_color'], 1)