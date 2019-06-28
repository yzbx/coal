import time
import json
import cv2
import os
from app.yolov3 import yolov3_detect
from app.bg_process import car_detection
def detection(data_json):
    data=json.loads(data_json)
    if data['task_name']=='car_detection':
        p=car_detection(data['video_url'])
        p.bg_process()
    else:
        for i in range(100):
            print(i,data)
            time.sleep(1)

    return 0
class video_buffer():
    def __init__(self,buffer_size=1200):
        self.buffer_size=buffer_size
        self.current_frame=-1
        self.buffer=[]

    def update(self,frame):
        self.current_frame+=1
        if self.current_frame>=self.buffer_size:
            self.current_frame=0

        self.buffer[self.current_frame]=frame

class video_saver():
    def __init__(self,video_buffer,save_seconds=600):
        self.video_buffer=video_buffer
        self.video_size=save_seconds*30
        self.key_frame=video_buffer.current_frame
        self.writer=None
        video_name=str(int(time.time()))+'.mp4'
        self.video_path=os.path.join('media',video_name)
        self.frame_number=0

        assert self.video_buffer.buffer_size>=self.video_size
        for idx in range(self.key_frame-self.video_size//2,self.key_frame+1):
            valid_idx=(idx+self.video_buffer.buffer_size)%self.video_buffer.buffer_size
            self.write(self.video_buffer.buffer[valid_idx])

    def write(self,frame):
        if self.writer is None:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            height,width=frame.shape[0:2]
            self.writer = cv2.VideoWriter(self.video_path, codec, fps, (width, height))

        self.writer.write(frame)
        self.frame_number+=1

        if self.frame_number>self.video_size:
            assert False

    def close(self):
        if self.writer is not None:
            self.writer.close()

class app_player():
    def __init__(self,video_url,app='yolov3',show_full_img=False):
        self.video_url=video_url
        self.app=app
        self.show_full_img=show_full_img

    def gen(self):
        if self.app=='yolov3':
            for img in yolov3_detect(self.video_url,self.show_full_img):
                ret, img = cv2.imencode('.jpg', img)
                frame=img.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            raise StopIteration

class video_player():
    def __init__(self,video_url):
        self.video=cv2.VideoCapture(video_url)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, img = cv2.imencode('.jpg', image)
        return img.tobytes()

    def gen(self):
        idx=0
        while idx<1000:
            frame=self.get_frame()
            idx+=1
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
