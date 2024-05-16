import cv2
from ultralytics import YOLO

src = '3.mp4'
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(src)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_height = 720
ratio = new_height / height
new_width = int(width * ratio)
hoi = int(new_height/2)

while (cap.isOpened()):
    # Read frames from the video stream
    ret, in_frame = cap.read()
    if not ret: break

    r_frame = cv2.resize(in_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    frame = r_frame[hoi-100:-200, :]

    results = model.track(
        frame,
        # stream=True,
        persist=True,
        verbose=False,
        show=False,
        vid_stride=4,
        classes=[0]
        )

    # annotated_frame = results[0].plot()

    for result in results:
        for bbox in result.boxes:
            bbox = bbox.xyxy.tolist()
            print(bbox)
            cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[0][2]), int(bbox[0][3])), (0, 255, 0), 2)

    cv2.imshow('Result', frame)
    cv2.imshow('Live', r_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()