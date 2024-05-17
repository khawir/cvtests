from fastapi import Depends, APIRouter, HTTPException
from sqlmodel import Session
from models import Site
from core import crud, utils
from sse_starlette.sse import EventSourceResponse
import cv2
import base64
import time
from ultralytics import YOLO

router = APIRouter()

s = 'rtsp://admin:hik@12345@172.23.16.55'

router = APIRouter()

model = YOLO("yolov8n.pt")

def generate_frames(s):

    video = cv2.VideoCapture(s)
    try:
        while True:
            success, frame = video.read()
            if not success:
                yield dict(event="NO_FRAMES", data="No frames available")
                continue
            else:
                frame = cv2.resize(frame, (640, 480))

                results = model.track(source=frame, 
                    persist=True, 
                    show=False,
                    classes=[0],
                    vid_stride=4,
                    stream=True,
                    verbose=False,
                    )
                
                for result in results:
                    # print(result)
                    for bbox in result.boxes:
                        bbox = bbox.xyxy.tolist()
                        cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[0][2]), int(bbox[0][3])), (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                frame = buffer.tobytes()
                frame_base64 = base64.b64encode(frame).decode()
                yield f"data:image/jpeg;base64,{frame_base64}"

                time.sleep(0.03)
    except Exception as e:
        print(f"Error: {e}")
        yield dict(event="STREAM_INTERRUPTED", data="Error Stream Interrupted")
    finally:
        video.release()

@router.get("/video_feed/{site_id}")
async def video_feed(*, 
                session: Session = Depends(utils.get_session), 
                site_id: int
                ):
    db_site = session.get(Site, site_id)
    if not db_site:
        raise HTTPException(status_code=404, detail="Site not found")
    print(db_site.in_url)
    return EventSourceResponse(generate_frames(db_site.in_url))
