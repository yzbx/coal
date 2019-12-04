# -*- coding: utf-8 -*-

import unittest
import json
from app.framework import QD_Detector,QD_Alerter,QD_Writer,QD_Upload
from model.yolov3.utils.utils import plot_one_box
import cv2
import os

class PeopleDetectionTest(unittest.TestCase):
    def test_people_detection(self):
        with open('config.json','r') as f:
            config=json.load(f)

        config['task_name']='person'
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

    def test_demo(self):
#        self.assertTrue(False)
        def is_overlap(rect1,rect2):
            """
            retc1: [x11,y11,x12,y12]
            rect2: [x21,y21,x22,y22]
            """
            w=min(rect1[2],rect2[2])-max(rect1[0],rect2[0])
            h=min(rect1[3],rect2[3])-max(rect1[1],rect2[1])

            if w<=0 or h <=0:
                return False
            else:
                return True



        videos=['templates/coal_person_demo002.mp4','templates/coal_person_demo004.mp4']
        areas=[[300,300,550,500],[300,350,550,500]]

        with open('config.json','r') as f:
            config=json.load(f)

        detector=QD_Detector(config)

        COLOR_AREA=(0,0,255)
        COLOR_ALARM=(0,0,255)
        COLOR_NORMAL=(0,255,0)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        fps=30
        for video_url,warning_area in zip(videos,areas):
            cap=cv2.VideoCapture(video_url)
            writer=None
            save_video_name=os.path.join(os.path.dirname(video_url),'out_'+os.path.basename(video_url))

            self.assertTrue(cap.isOpened(),msg='cannot open video {}'.format(video_url))
            while True:
                flag,frame=cap.read()
                if not flag:
                    break
                image,bboxes=detector.process(frame)
                bboxes=[b for b in bboxes if b['label']=='person']
                for bbox in bboxes:
                    if is_overlap(warning_area,bbox['bbox']):
                        color=COLOR_ALARM
                    else:
                        color=COLOR_NORMAL

                    plot_one_box(bbox['bbox'], frame, label=bbox['label']+' %0.2f'%(bbox['conf']), color=color)

                frame=cv2.rectangle(img=frame, pt1=tuple(warning_area[0:2]), pt2=tuple(warning_area[2:4]), color=COLOR_AREA, thickness=2)
                cv2.imshow('person detection',frame)
                key=cv2.waitKey(30)
                if key==ord('q'):
                    break

                if writer is None:
                    height,width=frame.shape[0:2]
                    os.makedirs(os.path.dirname(save_video_name),exist_ok=True)
                    writer=cv2.VideoWriter(save_video_name,
                                    codec, fps,
                                    (width, height))

                writer.write(frame)

            writer.release()

if __name__ == '__main__':
    unittest.main()