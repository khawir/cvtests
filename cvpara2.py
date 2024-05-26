from datetime import datetime
import json
import threading
import time
import requests
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
import cv2
from collections import defaultdict, deque
from deepface import DeepFace
import numpy as np
from shapely.geometry import LineString, Point


class Tracker:
    def __init__(self, site_id, roi1=600, roi2=900, reg_pts=None, sensitivity=2):
        self.site_id = site_id

        self.sensitivity = sensitivity
        self.track_history = defaultdict(list)
        self.count_ids = []
        self.visits = []
        self.frame_buffer = deque(maxlen=4)

        self.roi1 = roi1
        self.roi2 = roi2
        self.line_dist_thresh = 15
        self.reg_pts = [(10, 650), (2000, 800)] if reg_pts is None else reg_pts
        self.counting_region = LineString(self.reg_pts)

    def snap_on_in(self, results, frame):
        cv2.polylines(frame, [np.array(self.reg_pts, dtype=np.int32)], isClosed=True, color=(0, 230, 0), thickness=2)

        boxes = results.boxes.xywh.cpu()
        xyxys = results.boxes.xyxy.cpu()
        if results.boxes.id is None:
            return
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id, xyxy in zip(boxes, track_ids, xyxys):
            x, y, _, _ = box

            track_line = self.track_history[track_id]
            track_line.append((float(x), float(y)))
            if len(track_line) > 30:
                track_line.pop(0)

            points = np.hstack(track_line).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            
            y_checked = int(y)
            if y_checked in range(self.roi1, self.roi2):

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                if prev_position is not None and track_id not in self.count_ids:
                    distance = Point(track_line[-1]).distance(self.counting_region)
                    if distance < self.line_dist_thresh and track_id not in self.count_ids:
                        self.count_ids.append(track_id)
                        # print(self.count_ids)

                        now = datetime.now()
                        visit = {}

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            visit["site_id"] = self.site_id
                            visit["ts"] = time.time()
                            visit["date_in"] = now.strftime("%Y-%m-%d")
                            visit["time_in"] = now.strftime("%H:%M")

                            if self.visits:
                                diff = abs(visit["ts"] - self.visits[-1]["ts"])
                                visit["is_group"] = True if diff<self.sensitivity else False

                            pers = save_one_box(
                                xyxy, 
                                results.orig_img.copy(),
                                # Path(f"cls/{track_id}.jpg"), 
                                BGR=True,
                                save=False,
                                )                        
                            visit["is_female"] = self.get_gender(pers)

                            self.visits.append(visit)
                            jvisit = json.dumps(visit)
                            print(jvisit)                    

                        else:
                            if self.count_ids:
                                self.count_ids.pop(0)
                            visit["site_id"] = site_id                       
                            visit["time_out"] = now.strftime("%H:%M")
                            jvisit = json.dumps(visit)
                            print(jvisit)


    def get_gender(self, pers):
        pers_gs = DeepFace.analyze(
            pers,
            actions=['gender'],
            enforce_detection=False,
            detector_backend='yolov8',
            # expand_percentage=10,
            silent=True
        )
        
        if pers_gs:
            # return round((pers_gs[0]['gender']['Woman']),1)
            return True if pers_gs[0]['gender']['Woman'] > 20 else False
        return None
    
    def get_vector(self, pers):
        pers_vs = DeepFace.represent(
            pers,
            model_name='SFace',
            enforce_detection=False,
            detector_backend='yolov8',
            # expand_percentage=10,
        )

        if pers_vs:
            return pers_vs[0]['embedding']
        return ""

    


def run_tracker_in_thread_v1(src: str, model: str, site_id: int, tracker: Tracker):
    while True:
        try:
            for result in model.track(
                source=src,
                stream=True,
                persist=True,
                max_det=10,
                conf=0.3,
                # stream_buffer=True,
                classes=[0],
                # show=True,
                # show_labels=False,
                # line_width=1,
                verbose=False
            ):
                frame = result.plot()

                if(result.boxes):
                    tracker.snap_on_in(result, frame)
                else:
                    pass
                
                r_frame = cv2.resize(frame, (680, 420))
                cv2.imshow(f'Tracking_Stream {site_id}', r_frame)
                
                tracker.frame_buffer.append(r_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        except Exception as e:
            print("Error:", e)
            print("Attempting to reconnect...")
            time.sleep(5)  # Wait for 5 seconds before attempting to reconnect
            continue
        else:
            print("Stream ended.")
            break




src = 'http://127.0.0.1:8080'
model = YOLO('yolov8n.pt')
site_id = 2

tracker = Tracker(site_id=site_id)

run_tracker_in_thread_v1(src, model, site_id, tracker)

