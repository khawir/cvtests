from datetime import datetime
import json
import time
import requests
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
import cv2
from collections import defaultdict
from deepface import DeepFace

model = YOLO("yolov8n.pt")

track_history = defaultdict(list)
count_ids = []
visits = []
roi1=600
roi2=900
sensitivity=2,
post_in_ep='http://127.0.0.1:8000/in'
post_out_ep='http://127.0.0.1:8000/out'
token="bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJraGF3aXIiLCJpc19zdXBlcnVzZXIiOnRydWUsImV4cCI6MTcxNjI3MDc1OH0.bOUI0OOjjDG23_XmR3KbEngtj3f0gQXoqyAxOUHdQLQ"
site_id=2

# src = "3.mp4"
# src = "http://127.0.0.1:8080"
# print(2560 * 720 / 1920)

# s1 = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716316220&id=712839666379337728&c=3cffb6de2e&t=c75230bb20b920e2dfebae890457b6528ae9a2f693d8e83f62500787b2057ce7&ev=100"
# s2 = 'rtsp://admin:hik@12345@172.23.16.55'
s2 = '3.mp4'


def post_visit(visit, endpoint):
    response = requests.post(
        endpoint, 
        headers={"Authorization": token, "Content-Type": "application/json"},
        data=visit
        )
    
    if response.status_code == 200:
        print("visit posted")
        
def get_gender(pers):
    pers_gs = DeepFace.analyze(
        pers,
        actions = ['gender'],
        enforce_detection=False,
        detector_backend='yolov8',
        # expand_percentage=10,
        silent=True
        )
    
    if pers_gs:
        # return round((pers_gs[0]['gender']['Woman']),1)
        return True if pers_gs[0]['gender']['Woman']>20 else False
    return None

def get_vector(pers):
    pers_vs = DeepFace.represent(
        pers,
        model_name='SFace',
        enforce_detection=False,
        detector_backend='yolov8',
        # expand_percentage=10,
    )

    if pers_vs:
        # print(f"{len(pers_vs[0]['embedding'])}")
        return pers_vs[0]['embedding']
    return ""



def snap_on_in(result):
    # frame = results.plot()
    # cv2.rectangle(frame, (0, roi1), (width, roi2), color=(0, 230, 0), thickness=2)

    boxes = result.boxes.xywh.cpu()
    xyxys = result.boxes.xyxy.cpu()
    if result.boxes.id is None:
        return
    track_ids = result.boxes.id.int().cpu().tolist()

    for box, track_id, xyxy in zip(boxes, track_ids, xyxys):
        x, y, _, _ = box

        track = track_history[track_id]
        # track.append((float(x), float(y)))  # x, y center point
        track.append((x, y))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        # cv2.polylines(self.frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        y_checked = int(y)
        if y_checked in range(roi1, roi2):
            # _, y_prev = track[len(track)-1] if len(track)>1 else None
            _, y_prev = track_history[track_id][-2] if len(track_history[track_id]) > 1 else None
            # _, y_prev = track[len(track)-2] if len(track) > 1 else None

            if y_prev is not None and track_id not in count_ids:
                count_ids.append(track_id)
                now = datetime.now()
                guest = {}
                # print(self.count_ids)

                print(f"{y} : {y_prev}")

                if y_checked > int(y_prev):
                    # print(f"{track_id} : In @ {time.time()}")                        
                    guest["track_id"] = track_id
                    guest["ts"] = time.time()
                    guest["date_in"] = now.strftime("%Y-%m-%d")
                    guest["time_in"] = now.strftime("%H:%M")
                    # guest["time_out"] = guest["time_in"]

                    if visits:
                        diff = abs(guest["ts"] - visits[-1]["ts"])
                        guest["is_group"] = True if diff<sensitivity else False

                    pers = save_one_box(
                        xyxy, 
                        frame.copy(), 
                        # Path(f"cls/{track_id}.jpg"), 
                        BGR=True,
                        save=False,
                        )
                    
                    guest["is_female"] = get_gender(pers)
                    # guest["vector"] = get_vector(pers)
                    guest["site_id"] = site_id
                    guest["is_new"] = True
                    
                    visits.append(guest)
                    visit = json.dumps(guest)
                    print(visit)
                    # post_visit(visit, post_in_ep)
                    
                elif y_checked < int(y_prev):
                    if count_ids:
                        count_ids.pop(0)
                    # guest["track_id"] = track_id
                    guest["time_out"] = now.strftime("%H:%M")
                    guest["site_id"] = site_id
                    visit = json.dumps(guest)
                    # self.visit(visit, post_out_ep)
                    print(visit)


while True:
    try:
        for result in model.track(
            source=s2, 
            stream=True, 
            persist=True,
            max_det=10,
            conf=0.3,
            # stream_buffer=True,
            classes=[0],
            # show=True,
            verbose=False
        ):
            frame = result.plot()
            if(result.boxes):
                snap_on_in(result)
            else:
                pass

            r_frame = cv2.resize(frame, (960, 720) )
            cv2.imshow('frame', r_frame)
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




# for result in model.track(
#     source=src, 
#     stream=True, 
#     persist=True,
#     max_det=10,
#     conf=0.5,
#     stream_buffer=True,
#     classes=[0],
#     # show=True,
#     ):
#     frame = result.plot()
#     r_frame = cv2.resize(frame, (960, 720) )
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break


# results = model.track(
#     source=src, 
#     stream=True,
#     persist=True,
#     # half=True,
#     # device='cuda:0',
#     max_det=10,
#     # vid_stride=2,
#     stream_buffer=True,
#     classes=[0],
#     # conf=0.5
#     show=True,
#     # save_txt=True,
#     # save_conf=True,
#     # show_label=False,
#     # show_conf=False,
#     # show_boxes=False,
#     # line_width=1,
#     # verbose=False
#     )


# for r in results:
#     boxes = r.boxes

# while True:
#     results = next(results_gen, None)
#     frame = results.plot()
#     # scale_percent = 20
#     # width = int(frame.shape[1] * scale_percent / 100)
#     # height = int(frame.shape[0] * scale_percent / 100)
#     # dim = (width, height)
#     # resized = cv2.resize(frame, dim)
#     cv2.imshow('yolo', frame)








