import pandas as pd
import unittest

from datageneration.data_model import Tag
from datageneration.tags_to_imr import transform_tags_to_imr

'''
Execute it as follows: python -m unittest datageneration.tests.test_tag_to_imr
'''


class TestTag2ImrConverter(unittest.TestCase):
    def setUp(self):
        self.primary_key_table = pd.read_excel('datageneration/tests/data/Primary_Keys_test.xlsx', engine='openpyxl')

    def test_tag_to_imr(self):
        tags_str = 'route=bus, route= train, route= ferry, route=tram, route=trolleybus, route=subway'
        expected_tags = [{"or": [Tag(key='route', operator='=', value='bus'),
                                 Tag(key='route', operator='=', value='train'),
                                 Tag(key='route', operator='=', value='ferry'),
                                 Tag(key='route', operator='=', value='tram'),
                                 Tag(key='route', operator='=', value='trolleybus'),
                                 Tag(key='route', operator='=', value='subway'),
                                 ]}]
        predicted_tags = transform_tags_to_imr(tags_str)

        self.assertEqual(expected_tags, predicted_tags)

        tags_str = 'sport=athletics AND athletics=hammer_throw'
        expected_tags = [{"and": [Tag(key='sport', operator='=', value='athletics'),
                                  Tag(key='athletics', operator='=', value='hammer_throw')]}]

        predicted_tags = transform_tags_to_imr(tags_str)
        self.assertEqual(expected_tags, predicted_tags)

        tags_str = '["hov:lanes:forward:conditional"|"hov:lanes:backward:conditional"|"hov:lanes:forward"|"hov:lanes:backward"|"hov:lanes:conditional"|"hov:lanes"]=designated'
        expected_tags = [{
            "or": [Tag(key='hov:lanes:forward:conditional', operator='=', value='designated'),
                   Tag(key='hov:lanes:backward:conditional', operator='=', value='designated'),
                   Tag(key='hov:lanes:forward', operator='=', value='designated'),
                   Tag(key='hov:lanes:backward', operator='=', value='designated'),
                   Tag(key='hov:lanes:conditional', operator='=', value='designated'),
                   Tag(key='hov:lanes', operator='=', value='designated'),

                   ]}]

        predicted_tags = transform_tags_to_imr(tags_str)
        self.assertEqual(expected_tags, predicted_tags)

        tags_str = 'addr:street~***any***'
        expected_tags = [{
            "or": [Tag(key='addr:street', operator='~', value='***any***')]
        }]
        predicted_tags = transform_tags_to_imr(tags_str)
        self.assertEqual(expected_tags, predicted_tags)

if __name__ == '__main__':
    unittest.main()
