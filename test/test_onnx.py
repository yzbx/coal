# -*- coding: utf-8 -*-

import onnxruntime
import cv2
import sys
import numpy as np
from scipy.special import softmax

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def simple_preprocess(image,img_size):
    # Padded resize
    img=cv2.resize(image,tuple(img_size),interpolation=cv2.INTER_LINEAR)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img

if __name__ == '__main__':
    assert len(sys.argv)==3
    print('load onnx weight from {}'.format(sys.argv[1]))
    ort_session=onnxruntime.InferenceSession(sys.argv[1])
    print('load video from {}'.format(sys.argv[2]))
    cap=cv2.VideoCapture(sys.argv[2])

    if not cap.isOpened():
        assert False

    names=['normal','fire']
    while True:
        flag,frame=cap.read()
        if not flag:
            break

        img=np.expand_dims(simple_preprocess(frame,(224,224)),0)
        ort_inputs={ort_session.get_inputs()[0].name:img}
        result=softmax(np.squeeze(ort_session.run(None,ort_inputs)))

        text=names[np.argmax(result)]
        if text==names[1]:
            color=(0,0,255)
        else:
            color=(255,0,0)

        print(text,result)
        # convert image to [height width channel] format
        fontScale=max(1,frame.shape[1]//448)
        thickness=max(1,frame.shape[1]//112)
        frame=cv2.putText(frame, text+' %0.2f'%(max(result)) , (50,50), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness)

        cv2.imshow('fire detection',frame)
        key=cv2.waitKey(30)
        if key==ord('q'):
            break