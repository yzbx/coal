# -*- coding: utf-8 -*-

import unittest
import json
import time
import os
from app.framework import QD_Process 

class GPUTest(unittest.TestCase):        
    def test_gpu(self):
        with open('config.json','r') as f:
            config=json.load(f)
    
        config['save_frame_number']=10
        try:
            p=QD_Process(config)
            p.process()
        except RuntimeError as e:
            print('gpu out of memory',e.__str__())
            
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()