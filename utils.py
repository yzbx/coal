import time
import json
import cv2
def detection(data_json):
    data=json.loads(data_json)
    for i in range(100):
        print(i,data)
        time.sleep(1)

    return 0

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