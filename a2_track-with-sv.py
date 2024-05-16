import cv2
from ultralytics import YOLO
import supervision as sv

src = 'video4.mp4'
# src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1715575960&id=709734792474394624&c=3cffb6de2e&t=3efe00338e6f7347505662a61cf1de28f63d523bb78d4a371c72932525da9c3d&ev=100'

# def main():
model = YOLO("yolov8n.pt")
# result = model.track(source=0, show=True)

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

for result in model.track(source=src, 
                            show=False, 
                            stream=True):
    
    frame = result.orig_img
    detections = sv.Detections.from_ultralytics(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[(detections.class_id == 0)]

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, _, confidence, class_id, tracker_id
        in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    cv2.imshow("yolov8", frame)

    if(cv2.waitKey(30) == 27):
        break

# if __name__ == "__main__":
#     main()