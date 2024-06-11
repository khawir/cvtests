from datetime import datetime
import json
from threading import Thread
import cv2, time
from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import save_one_box
from pathlib import Path
from collections import defaultdict
# import numpy as np
from deepface import DeepFace
import requests


class ThreadedCamera(object):
    def __init__(self, 
                 site_id, 
                 src, 
                 fps=20,
                 sensitivity=2, 
                 buffer_size=2, 
                 disp_height=720, 
                 roi1=600, 
                 roi2=900,
                 post_in_ep='http://127.0.0.1:8000/in',
                 post_out_ep='http://127.0.0.1:8000/out',
                 token="bearer "
                 ):
        self.site_id = site_id
        self.capture = cv2.VideoCapture(src)
        self.fps = fps
        self.sensitivity = sensitivity
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_h = disp_height
        self.ratio = self.new_h/ self.height
        self.new_w = int(self.width * self.ratio)
        self.track_history = defaultdict(list)
        self.count_ids = []
        self.visits = []
        self.roi1 = roi1
        self.roi2 = roi2
        self.post_in_ep = post_in_ep
        self.post_out_ep = post_out_ep
        self.token = token

       
        self.FPS = 1/20
        self.FPS_MS = int(self.FPS * 1000)

        self.heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=True,
            shape="circle",
            classes_names=model.names
        )
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        tracks = model.track(
            self.frame,
            persist=True,
            max_det = 10,
            classes=[0],
            show=False,
            conf=0.1,
            # show_labels=False,
            verbose=False
        )

        r_frame = self.heatmap_obj.generate_heatmap(self.frame, tracks)

        # if results[0]:
        #     self.snap_on_in(results[0])
        # else:
        #     pass

        r_frame = cv2.resize(self.frame, (self.new_w, self.new_h) )
        cv2.imshow('frame', r_frame)
        if cv2.waitKey(self.FPS_MS) & 0xFF == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
    
            


# src = 'rtsp://admin:hik@12345@172.23.16.55'
# src = 'https://drive.google.com/file/d/14fTYfXCMoodYjW62k5rVA6Vaugc31MQ8/view?usp=drive_link'
# src = 'http://127.0.0.1:8080'

src = '3.mp4'   # 800, 900
# src = 'high.mp4'   # 800, 900
# 650, 900
# src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716122383&id=712026655901229056&c=3cffb6de2e&t=87e478fa2d77b5693906d36369f29208475c5a4a1c6882963814f2e066977922&ev=100'
model = YOLO('yolov8n.pt')

threaded_camera = ThreadedCamera(
    site_id=2,
    src=src,
    fps=20,
    sensitivity=2,
    buffer_size=2,
    disp_height=720,
    roi1=600,
    roi2=900,
    post_in_ep='http://127.0.0.1:8000/in',
    post_out_ep="http://127.0.0.1:8000/out",
    token="bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJraGF3aXIiLCJpc19zdXBlcnVzZXIiOnRydWUsImV4cCI6MTcxNjI3MDc1OH0.bOUI0OOjjDG23_XmR3KbEngtj3f0gQXoqyAxOUHdQLQ"
    )
while True:
    try:
        threaded_camera.show_frame()
    except AttributeError:
        pass