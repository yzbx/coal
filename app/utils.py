import cv2

def gen_imencode(gen):
    for img in gen:
        ret, img = cv2.imencode('.jpg', img)
        frame=img.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
