from threading import Thread
import cv2, time
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from pathlib import Path
from collections import defaultdict
import numpy as np
from deepface import DeepFace


class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_h = 720
        self.ratio = self.new_h/ self.height
        self.new_w = int(self.width * self.ratio)
        self.track_history = defaultdict(lambda: [])
        self.count_ids = []
        self.roi1 = 300
        self.roi2 = 400

       
        self.FPS = 1.0/20
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
            show_labels=False,
            verbose=False
        )

        self.snap_on_in(results[0])

        r_frame = cv2.resize(self.frame, (self.new_w, self.new_h) )
        cv2.imshow('frame', r_frame)
        if cv2.waitKey(self.FPS_MS) & 0xFF == ord('q'):
            self.capture.release()
            # self.thread.join()
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
                        pers = save_one_box(
                            xyxy, 
                            self.frame.copy(), 
                            Path(f"cls/{track_id}.jpg"), 
                            BGR=True,
                            # save=False
                            )
                        
                        pers_gs = DeepFace.analyze(
                            pers,
                            actions = ['gender'],
                            enforce_detection=False,
                            detector_backend='yolov8',
                            expand_percentage=25,
                            # silent=True
                            )
                        
                        print(f"{track_id} : In @ {time.time()}")
                        
                        if pers_gs:
                            for pers_g in pers_gs:
                                print(pers_g['dominant_gender'])
                        else:
                            print("no face detected")

                    else:
                        print(f"{track_id} : Out @ {time.time()}")
                    


# src = 'rtsp://admin:hik@12345@172.23.16.55'
src = '3.mp4'
# src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716039080&id=711677256444084224&c=3cffb6de2e&t=e1498a314974763a69bcfa322cfc66b560034821f1069a0ad4701eb74382c53c&ev=100'
model = YOLO('yolov8n.pt')

threaded_camera = ThreadedCamera(src)
while True:
    try:
        threaded_camera.show_frame()
    except AttributeError:
        pass