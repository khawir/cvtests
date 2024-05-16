import cv2
from deepface import DeepFace
import json

src = '3.mp4'
# src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1715575960&id=709734792474394624&c=3cffb6de2e&t=3efe00338e6f7347505662a61cf1de28f63d523bb78d4a371c72932525da9c3d&ev=100'

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
    frame = r_frame[hoi:, :]

    results = DeepFace.extract_faces(frame,
                                     enforce_detection=False,
                                     detector_backend='yolov8',
                                     )

    for result in results:
        x, y, w, h, _, _ = result["facial_area"].values()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    

    # demos = DeepFace.analyze(frame,
    #                          actions=['gender'],
    #                          detector_backend='yunet',
    #                          enforce_detection=False,
    #                          expand_percentage=10
    #                          )
   
    # print(json.dumps(demos))

    # for demo in demos:
    #     # print(f"{demo['dominant_gender']} : {demo['face_confidence']}")
    #     x, y, w, h, _, _ = demo["region"].values()
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Result', frame)
    cv2.imshow('Live', r_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()