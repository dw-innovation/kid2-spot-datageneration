import unittest
from unittest.mock import patch
from datageneration.data_model import Entity, Property, Relation
from datageneration.relation_generator import RelationGenerator

'''Run python -m unittest datageneration.tests.test_relation_generator'''


class TestRelationGenerator(unittest.TestCase):
    def setUp(self):
        self.relation_generator = RelationGenerator(max_distance_digits=5)

    def test_individual_distances(self):
        relations = self.relation_generator.generate_individual_distances(num_entities=3)
        dists = [r.value for r in relations]
        sources = [r.source for r in relations]
        targets = [r.target for r in relations]

        assert 1 <= len(set(dists)) <= len(relations)
        assert len(set(sources)) == len(relations)
        assert len(set(targets)) == len(relations)

    def test_within_radius(self):
        relations = self.relation_generator.generate_within_radius(num_entities=3)

        dists = [r.value for r in relations]
        sources = [r.source for r in relations]
        targets = [r.target for r in relations]

        assert len(set(dists)) == 1
        assert len(set(sources)) == 1
        assert len(set(targets)) == len(relations)

    def test_in_area(self):
        relations = self.relation_generator.generate_in_area(num_entities=1)

        assert relations == None

    def test_contain_rel(self):
        entities = [Entity(id=0, is_area=True, name='astro station', type='nwr', properties=[]),
                    Entity(id=1, is_area=False, name='block', type='nwr',
                           properties=[Property(name='height', operator='=', value='0.6 m')]),
                    Entity(id=2, is_area=False, name='scuba center', type='nwr', properties=[])]

        # case 1, expect no Exception
        try:
            relations = self.relation_generator.generate_relation_with_contain(entities=entities)
        except AssertionError:
            raise RuntimeError('The test should not be failed! Something is wrong.')

        # case 2, one area entity and the others are ...
        area_entity = Entity(id=0, is_area=True, name='astro station', type='nwr', properties=[])
        point_entities_connecting_to_area_entity = [Entity(id=1, is_area=False, name='block', type='nwr',
                                                           properties=[
                                                               Property(name='height', operator='=', value='0.6 m')]),
                                                    Entity(id=2, is_area=False, name='scuba center', type='nwr',
                                                           properties=[])]
        other_point_entities = []
        relations = self.relation_generator.generate_relation_with_contain_helper(area_entity=area_entity,
                                                                                  other_point_entities=other_point_entities,
                                                                                  point_entities_connecting_to_area_entity=point_entities_connecting_to_area_entity)
        expected_relations = [Relation(type='contains', source=0, target=1, value=None),
                              Relation(type='contains', source=0, target=2, value=None)]

        self.assertEqual(relations, expected_relations)

        # case 3, one area entity, only one point entity is in a contain relation, the other one is connected to others (individual distances)
        with patch.object(self.relation_generator, 'get_random_decimal_with_metric') as mock_get_random_decimal:
            # Set the return value of the mocked method
            mock_get_random_decimal.return_value = "100 m"

        area_entity = Entity(id=0, is_area=True, name='astro station', type='nwr', properties=[])
        point_entities_connecting_to_area_entity = [
            Entity(id=2, is_area=False, name='scuba center', type='nwr', properties=[])]
        other_point_entities = [Entity(id=1, is_area=False, name='block', type='nwr',
                                       properties=[Property(name='height', operator='=', value='0.6 m')])]
        relations = self.relation_generator.generate_relation_with_contain_helper(area_entity=area_entity,
                                                                                  other_point_entities=other_point_entities,
                                                                                  point_entities_connecting_to_area_entity=point_entities_connecting_to_area_entity)
        expected_relations = [Relation(type='contains', source=0, target=2, value=None),
                              Relation(type='dist', source=0, target=1, value="100 m"),
                              Relation(type='dist', source=2, target=1, value="100 m")]

        self.assertEqual(relations, expected_relations)
