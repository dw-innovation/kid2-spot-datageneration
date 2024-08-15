import unittest
from datageneration.enrich_geo_database import request_wd_id

'''Run python -m unittest datageneration.tests.enrich_geo_database'''


class TestAreaGenerator(unittest.TestCase):
    def setUp(self):
        pass

    def test_request_wd_id(self):
        wd = request_wd_id('Zvishavane District')
        self.assertEqual(wd, 'Q7505444')


