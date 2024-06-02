import threading

from datetime import datetime
from fastapi import Depends, APIRouter, HTTPException, Request
from sqlmodel import Session, select
from models import Guest, Site, User, Visit
from core import crud, utils
from collections import deque
from fastapi.responses import StreamingResponse
from collections import defaultdict
from ultralytics.utils.plotting import save_one_box
from deepface import DeepFace
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from typing import Annotated
from config import settings
from shapely.geometry import LineString, Point, Polygon
import numpy as np
import cv2
import time
import json

maximum_reconnect_attempts = settings.MAXIMUM_RECONNECT_ATTEMPTS
stream_frame_quality = settings.STREAM_FRAME_QUALITY
tracking_FPS = 20

router = APIRouter(tags=["Stream"])

class Tracker:
    def __init__(self, site_id, roi1, roi2, roi3, roi4, sensitivity,session, link, model, reg_pts=None, crop_frames = True):
        self.track_history = {}
        self.count_ids = []
        self.visits = []
        self.site_id = site_id
        self.roi1 = roi1
        self.roi2 = roi2
        self.roi3 = roi3
        self.roi4 = roi4
        self.sensitivity = sensitivity
        self.frame_buffer = deque(maxlen=4)
        self.stop_tracking = False
        self.session = session
        self.video_link = link
        self.model = model
        self.stop_tracking = False
        
        self.track_history = defaultdict(list)
        self.count_ids = []
        self.visits = []
        self.line_dist_thresh = 15
        self.reg_pts = [(roi3, roi1), (roi4, roi1), (roi4, roi2), (roi3, roi2)] if reg_pts is None else reg_pts
        # self.counting_region = LineString(self.reg_pts)
        self.counting_region = Polygon(self.reg_pts)
        self.reconnect_attempt_counter = 0
        self.maximum_reconnect_attempts = 3
        
        self.crop_frames = crop_frames
        
        self.start_x = 1180
        self.start_y = 0
        self.end_x = 2000
        self.end_y = 600
        
        self.video = cv2.VideoCapture(self.video_link)
        
        if self.video.isOpened():
            LOGGER.info(f"Successfully connected to {self.video_link} ✅")
                
        self.tracker_thread = threading.Thread(target=self.run_tracker_in_thread_v1, args=())
        self.tracker_thread.start()

    def run_tracker_in_thread_v1(self):
        print("Thread Started")
        
        while not self.stop_tracking and self.video.isOpened():
            try:
                if cv2.waitKey(1) == ord('q'):
                        break
                self.video.grab()
                success, frame = self.video.retrieve()
                
                if not success:
                    LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                    self.video.open(self.video_link)  # re-open stream if signal was lost
                
                if frame is None:
                    continue
                
                source_frame = frame
                if self.crop_frames:
                    source_frame = frame[self.start_y:self.end_y, self.start_x:self.end_x]
                    
                height, width = source_frame.shape[:2]
                angle = 5  # Change this angle as needed

                rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                source_frame = cv2.warpAffine(source_frame, rotation_matrix, (width, height))
                    
                detections = self.model.track(
                    source=source_frame,
                    stream=True,
                    persist=True,
                    max_det=10,
                    conf=0.3,
                    # stream_buffer=True,
                    classes=[0],
                    # show=True,
                    show_labels=False,
                    line_width=1,
                    verbose=False
                )
                for result in detections:
                    r_frame = result.plot()
                    
                    if(result.boxes):
                        self.snap_on_in(result, r_frame)
                    else:
                        pass
                    
                    self.reconnect_attempt_counter = 0
                    # self.frame_buffer.append(frame)
                    self.frame_buffer.append(r_frame)

                    
                    cv2.imshow(f'Tracking_Stream_{self.site_id}', r_frame)
            except Exception as e:
                print(str(e))
                if self.reconnect_attempt_counter > self.maximum_reconnect_attempts:
                    break
                else:
                    self.attempt_reconnect()
                    
        self.stop_tracking = True
        del utils.trackers[self.site_id]
        cv2.destroyWindow(f'Tracking_Stream_{self.site_id}')
        
    def attempt_reconnect(self):
        print("Waiting...")
        time.sleep(3)
        print("Attempting to reconnect...")
        self.video.release()
        print("Restarting video capture...")
        self.video = cv2.VideoCapture(self.video_link)
        self.reconnect_attempt_counter += 1
        
        if self.video.isOpened():
            LOGGER.info(f"Successfully connected to {self.video_link} ✅")

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
            track_line.append((float(x), float(y)))  # x, y center point
            if len(track_line) > 30:  # retain 30 tracks for 30 frames
                track_line.pop(0)
                
            points = np.hstack(track_line).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            y_checked = int(y)
            # if y_checked in range(self.roi1, self.roi2):
            if self.counting_region.contains(Point(track_line[-1])):
                xy_prev = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                if xy_prev is not None and track_id not in self.count_ids:
                    self.count_ids.append(track_id)
                    now = datetime.now()
                    
                    visit = {}

                    if abs(self.roi1 - y_checked) < abs(self.roi2 - y_checked):
                        visit["site_id"] = self.site_id
                        visit["ts"] = time.time()
                        visit["date_in"] = now.strftime("%Y-%m-%d")
                        visit["time_in"] = now.strftime("%H:%M")

                        if self.visits:
                            diff = abs(visit["ts"] - self.visits[-1]["ts"])
                            visit["is_group"] = True if diff < self.sensitivity else False

                        pers = save_one_box(
                            xyxy, 
                            results.orig_img.copy(),
                            # Path(f"cls/{track_id}.jpg"), 
                            BGR=True,
                            save=False,
                        )                        
                        visit["is_female"] = self.get_gender(pers)
                        print(visit)
                        self.visits.append(visit)
                        db_visit = Visit(**visit)
                        self.session.add(db_visit)
                        self.session.commit()

                        visit_vector = self.get_vector(pers)                        
                        v0 = np.array(visit_vector)
                        db_data = self.session.exec(select(Guest.id, Guest.vector)).all()
                        if db_data:
                            dists = (0, 2)
                            for db_id, db_vector in db_data:
                                v = [float(x) for x in db_vector.split(",")]
                                v1 = np.array(v)

                                if len(v0) == len(v1):
                                    a = np.matmul(np.transpose(v0), v1)
                                    b = np.sum(np.multiply(v0, v0))
                                    c = np.sum(np.multiply(v1, v1))
                                    dist = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
                                    print("="*50)
                                    print(f"dist with {db_id}: {dist}")
                                    
                                    if dist < dists[1]:
                                        dists = (db_id, dist)

                            if dists[1] <= 0.593:
                                match = dists[0]
                                print("="*50)
                                print(f"match found at id {match}")
                                extra_in_data = {"guest_id": match,
                                            "is_new": False}
                                db_visit.sqlmodel_update(db_visit, update=extra_in_data)
                                self.session.add(db_visit)
                                self.session.commit()
                            else:
                                print("="*50)
                                print(f"adding new guest entry: {track_id}")                     
                                guest = {}
                                guest["name"] = track_id
                                guest["vector"] = ",".join(str(x) for x in visit_vector)
                                guest["site_id"] = self.site_id
                                db_guest = Guest(**guest)
                                self.session.add(db_guest)
                                self.session.commit()
                        
                    else:
                        if self.count_ids:
                            self.count_ids.pop(0)
                        now = datetime.now()
                        today = now.strftime("%Y-%m-%d")
                        query = select(Visit).filter(Visit.site_id == self.site_id, Visit.date_in==today, Visit.time_out == None).order_by(Visit.time_in)
                        fifo_visit = self.session.exec(query).first()
                        if fifo_visit:
                            print(f"Time_out Update on : {fifo_visit}")
                            extra_out_data = {"time_out": now.strftime("%H:%M")}
                            fifo_visit.sqlmodel_update(fifo_visit, update=extra_out_data)
                            self.session.add(fifo_visit)
                            self.session.commit()
    
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

async def create_stream(tracker: Tracker, request: Request):
    try:
        while not tracker.stop_tracking:
            if await request.is_disconnected():
                break
            
            if tracker.stop_tracking:
                break
            
            if tracker.frame_buffer:
                frame = tracker.frame_buffer[-1]
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if not ret:
                    continue
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.05)
        
        print("streaming loop ended.")
    except Exception as e:
        print(f"Error in frame streaming: {e}")

@router.get("/video_feed/{site_id}")
async def video_feed(site_id: int, 
                    request: Request,
                    session: Session = Depends(utils.get_session)
                    ):
    if site_id in utils.trackers:
        return StreamingResponse(create_stream(utils.trackers[site_id], request),
            media_type="multipart/x-mixed-replace; boundary=frame")
        
    else:
        db_site = session.get(Site, site_id)
        if not db_site:
            raise HTTPException(status_code=404, detail="Site not found")
        
        if(db_site.in_url):
            model = YOLO("yolov8n.pt")
            tracker = Tracker(site_id=site_id, roi1=200, roi2=300, roi3=170, roi4=510, sensitivity=2, session=session, model=model, link=db_site.in_url)

            utils.trackers[site_id] = tracker
            return StreamingResponse(create_stream(tracker, request),
                media_type="multipart/x-mixed-replace; boundary=frame")

@router.post("/stop_tracking/{site_id}")
def stop_stream(*,
                site_id: int,
                session: Session = Depends(utils.get_session),
                current_user: Annotated[User, Depends(crud.get_current_super_user)]
                ):
    db_site = crud.get_current_site(session, current_user, site_id)
    if db_site:
        try:
            utils.trackers[db_site.id].stop_tracking = True
            utils.trackers[db_site.id].tracker_thread.join()

            return {"message": "Site Tracking Closed!"}
        except:
            raise HTTPException(status_code=404, detail="model already stopped.")
    else:
        raise HTTPException(status_code=404, detail="Site not found.")
    

