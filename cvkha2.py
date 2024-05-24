import collections
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")


src = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716316220&id=712839666379337728&c=3cffb6de2e&t=c75230bb20b920e2dfebae890457b6528ae9a2f693d8e83f62500787b2057ce7&ev=100"

prev_frame = None

for result in model.track(
    source=src,
    stream=True,
    persist=True,
    max_det=10,
    conf=0.5,
    stream_buffer=True,
    classes=[0],
    # show=True,
):
    frame = result.plot()

    # Handle missing frames
    if prev_frame is None:
        prev_frame = frame.copy()
    elif frame is None:  # Missing frame detected
        interpolated_frame = cv2.warpAffine(prev_frame, 
                                            np.eye(2, 3), 
                                            (frame.shape[1], frame.shape[0]))
        prev_frame = frame.copy() if frame is not None else interpolated_frame  # Update for next frame
        frame = interpolated_frame
    else:
        prev_frame = frame.copy()

    r_frame = cv2.resize(frame, (960, 720))
    cv2.imshow('frame', r_frame)
    if cv2.waitKey(1) == ord('q'):
        break
