# -*- coding: utf-8 -*-

import unittest
import json
import time
import os

class DatabaseTest(unittest.TestCase):
    def test_insert(self):
        from app.framework import QD_Database
        with open('config.json','r') as f:
            config=json.load(f)
            
        d=QD_Database(config)
        content='test insert'
        id=d.insert(content)
        
        result=d.query(id)
        self.assertTrue(result.content==content)
        
    def test_update(self):
        from app.framework import QD_Database
        with open('config.json','r') as f:
            config=json.load(f)
            
        d=QD_Database(config)
        content='test update'
        id=d.insert(content)
        
        fileUrl='www.baidu.com'
        d.update(id,fileUrl)
        
        result=d.query(id)
        self.assertTrue(result.fileUrl==fileUrl)   
        
if __name__ == '__main__':
    unittest.main()
