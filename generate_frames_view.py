import cv2
import os
import pandas as pd

from flask import Blueprint, Response
from ultralytics import YOLO
from datetime import datetime
from Pulsse.database import db
from Pulsse.models.visits import Visits
from Pulsse import pulsse_app

frames_blueprint = Blueprint('/', __name__, url_prefix='/')

def gen_frames(model, link, key):
    
    results_generator = model.track(source=link, classes=0, stream=True, persist=True, conf=0.5)
    while True:
        results = next(results_generator, None)
        frame = results.plot()
        scale_percent = 20
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', resized)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(buffer) + b'\r\n')



@frames_blueprint.route('video_feed/<int:key>', methods=['GET'])
def video_feed(key):
    if key== 1:
        videolink = "https://isgpopen.ezvizlife.com/v3/openlive/AB9438217_1_1.m3u8?expire=1712085807&id=695096027447398400&c=a38fde48c5&t=07a21f8ab64be4a950b2853ab4ae87edf5b3517dddd59c40a77c76f56aa07274&ev=100"
        
    if key==2:
        videolink = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1712085807&id=695096030577836032&c=3cffb6de2e&t=1f36e17abe6be1e65ac2ec83c9a83cbb79760872ebd0ecde0afbdca335b9bb88&ev=100"
        
    # if key==3:
        # videolink = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1712085807&id=695096030577836032&c=3cffb6de2e&t=1f36e17abe6be1e65ac2ec83c9a83cbb79760872ebd0ecde0afbdca335b9bb88&ev=100"
        #videolink = "C:\\Users\\ahare\\Downloads\\Untitled video - Made with Clipchamp.mp4"
    # cap = cv2.VideoCapture(videolink)
    # print(cap)
    # success, frame = cap.read()
    # print(success)
    # fps = cap.get(5)
    # print(fps)

    model = YOLO('ml_models/yolov8n.pt')

    return Response(gen_frames(model, videolink, key), mimetype='multipart/x-mixed-replace; boundary=frame')