import time
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

s1 = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716316220&id=712839666379337728&c=3cffb6de2e&t=c75230bb20b920e2dfebae890457b6528ae9a2f693d8e83f62500787b2057ce7&ev=100"
s2 = 'rtsp://admin:hik@12345@172.23.16.55'

src = [s1, s2]


while True:
    try:
        for results in model.track(
            source="file.streams", 
            stream=True, 
            persist=True,
            max_det=10,
            # conf=0.4,
            # stream_buffer=True,
            classes=[0],
            # show=True,
            verbose=False
        ):
            # pass
            for result in results:
                frame = result.plot()
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