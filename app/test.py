# -*- coding: utf-8 -*-

import unittest
import json
import os

class TestMethods(unittest.TestCase):
    def test_json(self):
        cfg={'host':"10.0.0.39",
          'user':"iscas",
          'passwd':"sketch_123",
          'port':8306,
          'database':'qingdao',}
        
        if not os.path.exists('config.json'):
            with open('config.json','r') as f:
                json.dump(cfg,f)
                f.close()
        self.assertTrue(True)
        
    def test_database(self):
        import mysql.connector
        with open('config.json','r') as f:
            config=json.load(f)
            f.close()
            
        mydb = mysql.connector.connect(
          host=config['host'],
          user=config['user'],
          passwd=config['passwd'],
          port=config['port'],
          database=config['database'],
        )

        mycursor = mydb.cursor()

        mycursor.execute("SHOW DATABASES")
        
        databases=[]
        print('database','*'*30)
        for x in mycursor:
          print(x)
          databases.append(x)
        
        mycursor.close()
        self.assertTrue(len(databases)>0)
    
    def test_table(self):
        import mysql.connector
        
        with open('config.json','r') as f:
            config=json.load(f)
            f.close()
        mydb = mysql.connector.connect(
          host=config['host'],
          user=config['user'],
          passwd=config['passwd'],
          port=config['port'],
          database=config['database'],
        )

        mycursor = mydb.cursor()

        mycursor.execute("SHOW TABLES")
        print('table','*'*30)
        tables=[]
        for x in mycursor:
          print(x)
          tables.append(x)
        
        mycursor.close()
        self.assertTrue(len(tables)>0)
        
    def test_columns(self):
        import mysql.connector
        with open('config.json','r') as f:
            config=json.load(f)
            f.close()
        mydb = mysql.connector.connect(
          host=config['host'],
          user=config['user'],
          passwd=config['passwd'],
          port=config['port'],
          database=config['database'],
        )

        mycursor = mydb.cursor()
        
        results=[]
        mycursor.execute("show columns from mtrp_alarm")
        print('mtrp_alarm','*'*30)
        
        for x in mycursor:
          print(x)
          results.append(x)
          
        mycursor.execute("show columns from mtrp_alarm_type")
        print('mtrp_alarm_type','*'*30)
        for x in mycursor:
          print(x)
          results.append(x)
        
        mycursor.close()
        self.assertTrue(len(results)>0)
        
    def test_upload(self):
        import requests
        url='http://10.50.200.171:8080/mtrp/file/json/upload.jhtml'
        
        with open('test.png','rb') as f:
            files = {'file': f}
            r = requests.post(url, files=files)
            print('upload','*'*30)
            print(r.content)
            r.close()
            f.close()
        
        self.assertTrue(r.status_code==200 and r.ok)
        
    def test_sqlalchemy(self):
        from sqlalchemy import create_engine
        from sqlalchemy.sql import select
        from database import metadata,mtrp_alarm,mtrp_alarm_type
#        engine = create_engine('sqlite:///sqlite.db', echo=True)
        with open('config.json','r') as f:
            config=json.load(f)
            f.close()
            
        engine = create_engine('mysql://{}:{}@{}:{}/{}'.format(config['user'],
                               config['passwd'],
                               config['host'],
                               config['port'],
                               config['database']),
                               echo=True)
        conn=engine.connect()
        # This will check for the presence of each table first before creating, so it’s safe to call multiple times:
        metadata.create_all(engine)
        
        print('select mtrp_alarm','*'*30)
        s=select([mtrp_alarm.c.id,mtrp_alarm.c.pic_name,mtrp_alarm.c.pic_url])
        result=conn.execute(s)
        for row in result:
            print(row)
        
        result.close()
        
        print('select mtrp_alarm_type','*'*30)
        s=select([mtrp_alarm_type.c.id,mtrp_alarm_type.c.event_name])
        result=conn.execute(s)
        for row in result:
            print(row)
        
        result.close()
        self.assertTrue(True)
        
    def test_mysql_connector(self):
        import mysql.connector
        with open('config.json','r') as f:
            config=json.load(f)
            f.close()
            
        mydb = mysql.connector.connect(
          host=config['host'],
          user=config['user'],
          passwd=config['passwd'],
          port=config['port'],
        )
        print(mydb)
        mydb.close()
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()