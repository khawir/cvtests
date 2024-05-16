from threading import Thread
import cv2, time
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from collections import defaultdict
import numpy as np
# from deepface import DeepFace


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
        
        
        self.line_points = [(20, 400), (1080, 400)]
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=False,
                 reg_pts=self.line_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)


       
        self.FPS = 1/30
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
            # max_det = 10,
            classes=[0],
            show=False,
            conf=0.5,
            verbose=False
        )

        # self.snap_on_in(results[0])
        self.frame = self.counter.start_counting(self.frame, results)
        print(self.frame)

        r_frame = cv2.resize(self.frame, (self.new_w, self.new_h) )
        cv2.imshow('frame', r_frame)
        cv2.waitKey(self.FPS_MS)

    def snap_on_in(self, results):
        self.frame = results.plot()

        boxes = results.boxes.xywh.cpu()
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, _, _ = box
            track_line = self.track_history[track_id]
            track_line.append((float(x), float(y)))  # x, y center point
            if len(track_line) > 30:  # retain 90 tracks for 90 frames
                track_line.pop(0)

            # Draw the tracking lines
            points = np.hstack(track_line).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # print(f"{track_id} : {points.shape}")
            # if track_id==3:
            #     print(f"{track_id} : {y=}")
            y_checked = int(y)
            if y_checked in range(800,900):
                # print(f"{track_id} : Crossed @ {time.time()} | {len(track)}")
                _, y_prev = track_line[len(track_line)-2]
                print(f"{y} | {y_prev}")
                if y > y_prev:
                    print(f"{track_id} : In @ {time.time()}")
                else:
                    print(f"{track_id} : Out @ {time.time()}")
                    


# src = 'rtsp://admin:hik@12345@172.23.16.55'
src = '3.mp4'
# src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1715948858&id=711298838304219136&c=3cffb6de2e&t=15afff4112c95a983d9538ed19efc996e65feb1751b4b97105863b96bf14dfbc&ev=100'
model = YOLO('yolov8n.pt')

threaded_camera = ThreadedCamera(src)
while True:
    try:
        threaded_camera.show_frame()
    except AttributeError:
        pass