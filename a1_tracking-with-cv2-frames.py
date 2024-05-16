from ultralytics import YOLO
import cv2

model = YOLO(
    # 'yolov8n.pt'
    'yolov8n-face.pt'
    )

src = 'video4.mp4'
# src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1715575960&id=709734792474394624&c=3cffb6de2e&t=3efe00338e6f7347505662a61cf1de28f63d523bb78d4a371c72932525da9c3d&ev=100'
cap = cv2.VideoCapture(src)

ret = True
while ret:
    ret, frame = cap.read()
    results = model.track(frame, 
                          persist=True,
                        #   classes=[0]
                          )

    if ret:
        frame_ = results[0].plot()

        cv2.imshow('frame', frame_)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break