from datetime import datetime, timedelta
import json
from threading import Thread
import cv2, time
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from pathlib import Path
from collections import defaultdict
import numpy as np
from deepface import DeepFace


class ThreadedCamera(object):
    def __init__(self, 
                 site_id, 
                 src, 
                 fps=20,
                 sensitivity=2, 
                 buffer_size=2, 
                 disp_height=720, 
                 roi1=600, 
                 roi2=900
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

       
        self.FPS = 1/20
        self.FPS_MS = int(self.FPS * 1000)
        
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
        results = model.track(
            self.frame,
            persist=True,
            max_det = 10,
            classes=[0],
            show=False,
            conf=0.5,
            # show_labels=False,
            verbose=False
        )

        if results[0]:
            self.snap_on_in(results[0])
        else:
            pass

        r_frame = cv2.resize(self.frame, (self.new_w, self.new_h) )
        cv2.imshow('frame', r_frame)
        if cv2.waitKey(self.FPS_MS) & 0xFF == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def snap_on_in(self, results):
        # self.frame = results.plot()
        # cv2.rectangle(self.frame, (0, self.roi1), (self.width, self.roi2), color=(0, 230, 0), thickness=2)

        boxes = results.boxes.xywh.cpu()
        xyxys = results.boxes.xyxy.cpu()
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id, xyxy in zip(boxes, track_ids, xyxys):
            x, y, _, _ = box

            track = self.track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(self.frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            y_checked = int(y)
            if y_checked in range(self.roi1, self.roi2):
                # _, y_prev = track[len(track)-1]
                # _, y_prev = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                xy_prev = track[len(track)-2] if len(track)>1 else None

                if xy_prev is not None and track_id not in self.count_ids:
                    self.count_ids.append(track_id)

                    if y > xy_prev[1]:
                        # print(f"{track_id} : In @ {time.time()}")
                        guest = {}
                        guest["ts"] = time.time()
                        now = datetime.now()
                        guest["date_in"] = now.strftime("%Y-%m-%d")
                        guest["time_in"] = now.strftime("%H:%M")
                        
                        if self.visits:
                            diff = abs(guest["ts"] - self.visits[-1]["ts"])
                            guest["is_group"] = True if diff<self.sensitivity else False

                        pers = save_one_box(
                            xyxy, 
                            self.frame.copy(), 
                            Path(f"cls/{track_id}.jpg"), 
                            BGR=True,
                            save=False,
                            )
                        
                        guest["is_female"] = self.get_gender(pers)
                        # guest["vector"] = self.get_vector(pers)
                        guest["site_id"] = self.site_id
                        
                        self.visits.append(guest)
                        visit = json.dumps(guest)
                        print(visit)
                        
                    else:
                        print(f"{track_id} : Out @ {time.time()}")

        
    def get_gender(self, pers):
        pers_gs = DeepFace.analyze(
            pers,
            actions = ['gender'],
            enforce_detection=False,
            detector_backend='yolov8',
            expand_percentage=10,
            silent=True
            )
        
        if pers_gs:
            # print(f"{pers_gs[0]['dominant_gender']}")
            return True if pers_gs[0]['dominant_gender']=="Woman" else False
        return None

    def get_vector(self, pers):
        pers_vs = DeepFace.represent(
            pers,
            model_name='Dlib',
            enforce_detection=False,
            detector_backend='yolov8',
            expand_percentage=10,
        )

        if pers_vs:
            # print(f"{len(pers_vs[0]['embedding'])}")
            return pers_vs[0]['embedding']
        return ""
        


# src = 'rtsp://admin:hik@12345@172.23.16.55'
src = '3.mp4'   # 600, 900
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
    )
while True:
    try:
        threaded_camera.show_frame()
    except AttributeError:
        pass