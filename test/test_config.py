# -*- coding: utf-8 -*-

import unittest
import json
import os
from easydict import EasyDict as edict

class ConfigTest(unittest.TestCase):  
    def test_config(self):
        with open('config.json','r') as f:
            config=edict(json.load(f))
        
        for model in config.models:
            self.assertTrue(os.path.exists(model.cfg),
                            'cfg file {} not exist'.format(model.cfg))
            self.assertTrue(os.path.exists(model.data_cfg),
                            'data file {} not exist'.format(model.data_cfg))
            self.assertTrue(os.path.exists(model.weights),
                            'weight file {} not exist'.format(model.weights))
        
if __name__ == '__main__':
    unittest.main()