# -*- coding: utf-8 -*-

import unittest
import json
import time
import os

class UtilTest(unittest.TestCase):        
    def test_multi_process(self):
        from multiprocessing import Process
        def fun(note):
            for i in range(5):
                time.sleep(1)
                print("note: {}".format(note),i)
        
        
        p1=Process(target=fun,args=('p1 detection car',))
        print('start p1')
        p1.start()
        p2=Process(target=fun,args=('p2 detection helmet',))
        print('start p2')
        p2.start()
        print('join p1')
        p1.join()
        print('join p2')
        p2.join()
        
        self.assertTrue(True)
        
    def test_queue(self):
        from multiprocessing import Queue,Process
        import time
        import datetime
        import numpy as np
        
        def ask_time(q):
            start=datetime.datetime.now()
            while True:
                now=datetime.datetime.now()
                delta=now-start
                s=delta.seconds
                a=np.random.randint(low=0,high=10,size=(2,3))
                time.sleep(0.1)
                q.put((s,a))
        q=Queue()
        p=Process(target=ask_time,args=(q,))
        p.start()
        
        for i in range(10):
            print(q.get(),q.qsize())
            time.sleep(1)
        
        p.terminate()
        p.join()
        self.assertTrue(q.qsize()>=80)
        
if __name__ == '__main__':
    unittest.main()
