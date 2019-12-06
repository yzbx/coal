# -*- coding: utf-8 -*-

import unittest
import json
from app.framework import QD_Detector,QD_Alerter,QD_Writer,QD_Upload
import cv2
import os
import glob

class FireDetectionTest(unittest.TestCase):
    def test_people_detection(self):
        with open('config.json','r') as f:
            config=json.load(f)

        video_url=config['video_url']
        detector=QD_Detector(config)

        cap=cv2.VideoCapture(video_url)

        self.assertTrue(cap.isOpened(),msg='cannot open video {}'.format(video_url))

        alert=QD_Alerter(config)
        uploader=QD_Upload()
        filenames=[str(i)+'.jpg' for i in range(config['save_frame_number'])]
        for i in range(3):
            flag,frame=cap.read()
            self.assertTrue(flag,msg='cannot read frame from video {}'.format(video_url))

            image,bbox=detector.process(frame)

            save_image_file=os.path.join('static',filenames[i])
            cv2.imwrite(save_image_file,image)
            file_url=uploader.upload(save_image_file)
            print(file_url)

            rule=alert.bbox2rule(bbox)

            data_dict={'content':rule,'bbox':json.dumps(bbox)}
            if rule:
                writer=QD_Writer(config,rule,filenames)
                writer.insert_database(data_dict)
                writer.update_database((file_url,file_url))
                print('insert database rule is {}'.format(rule))

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()