import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np

LINE_START = sv.Point(0, 300)
LINE_END = sv.Point(960, 300)


line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

src = 'video4.mp4'
# src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1715575960&id=709734792474394624&c=3cffb6de2e&t=3efe00338e6f7347505662a61cf1de28f63d523bb78d4a371c72932525da9c3d&ev=100'

model = YOLO("yolov8n.pt")

for result in model.track(source=src, 
    show=False, 
    stream=True, 
    agnostic_nms=True,
    classes=[0]
    ):
    
    frame = result.orig_img
    detections = sv.Detections.from_ultralytics(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
    
    # detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, _, confidence, class_id, tracker_id
        in detections
    ]

    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections,
        labels=labels
    )

    line_counter.trigger(detections=detections)
    line_annotator.annotate(frame=frame, line_counter=line_counter)

    cv2.imshow("yolov8", frame)

    if (cv2.waitKey(30) == 27):
        break