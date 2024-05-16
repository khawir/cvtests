from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import time
from ultralytics import YOLO

s='https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8'

app = FastAPI()
cap = cv2.VideoCapture()
model = YOLO("yolov8n.pt")


# generator
def test_track():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model.track(source=frame, 
                              persist=True, 
                              show=False,
                              classes=[0],
                              vid_stride=15,
                              stream=True
                              )
        
        for result in results:
            print(result)
            for bbox in result.boxes:
                bbox = bbox.xyxy.tolist()
                cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[0][2]), int(bbox[0][3])), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)


@app.get("/tracked_video_feed")
async def tracked_video_feed():
    return StreamingResponse(test_track(), media_type='multipart/x-mixed-replace; boundary=frame')
