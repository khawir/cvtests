import threading
from fastapi import Depends, APIRouter, HTTPException
# from fastapi.responses import StreamingResponse
from sqlmodel import Session
from models import Site
from core import utils
import cv2
import time
from ultralytics import YOLO
import queue

#
from typing import Mapping, Union
import cv2
import time
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
#

router = APIRouter()

model = YOLO("yolov8n.pt")
default_frame_frequency = 15

# class ThreadedCamera:
#     def __init__(self, rtsp_url: str):
#         self.capture = cv2.VideoCapture(rtsp_url)
#         self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

#         if not self.capture.isOpened():
#             raise RuntimeError(f"Failed to open video stream: {rtsp_url}")

#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()

#         self.status = False
#         self.frame = None

#     def update(self):
#         while True:
#             if self.capture.isOpened():
#                 (self.status, self.frame) = self.capture.read()

#     def grab_frame(self):
#         if self.status:
#             return self.frame
#         return None

# class FrameStreamer:
#     def __init__(self, rtsp_url: str):
#         self.rtsp_url = rtsp_url
#         self.threaded_camera = ThreadedCamera(rtsp_url)

#     def _start_stream(self, freq: int = default_frame_frequency):
#         sleep_duration = 1.0 / freq

#         while True:
#             time.sleep(sleep_duration)
#             frame = self.threaded_camera.grab_frame()
#             if frame is None:
#                 continue

#             frame = cv2.resize(frame, (680, 320))
#             (flag, encodedImage) = cv2.imencode(".jpg", frame)
#             if not flag:
#                 continue
#             yield (b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

#     def get_stream(self, freq: int = default_frame_frequency, status_code: int = 206,
#                    headers: Union[Mapping[str, str], None] = None,
#                    background: Union[BackgroundTasks, None] = None) -> StreamingResponse:

#         return StreamingResponse(self._start_stream(freq),
#                                  media_type="multipart/x-mixed-replace;boundary=frame",
#                                  status_code=status_code,
#                                  headers=headers,
#                                  background=background)

# @router.get("/video_feed/{site_id}")
# async def video_feed(*,
#                      session: Session = Depends(utils.get_session),
#                      site_id: int):
#     db_site = session.get(Site, site_id)
#     if not db_site:
#         raise HTTPException(status_code=404, detail="Site not found")
#     print(db_site.in_url)
#     streamer = FrameStreamer("rtsp://admin:hik@12345@172.23.16.55")
#     return streamer.get_stream()


# class FrameStreamer:
#     #Defining constructor
#     def __init__(self, rtsp_url: str):
#         self.rtsp_url = rtsp_url

#         self.vs = None

#         try:
#             self.vs = cv2.VideoCapture(self.rtsp_url)
#             self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 2)
#             if not self.vs.isOpened():
#                 raise RuntimeError(f"Failed to open video stream: {rtsp_url}")
#         except Exception as e:
#             print(f"Error initializing video capture: {e}")

#     def _start_stream(self, freq: int = default_frame_frequency):
#         sleep_duration = 1.0 / freq

#         while True:
#             cv2.waitKey(75)
#             ret, frame = self.vs.read()
#             if not ret or frame is None:
#                 continue

#             frame = cv2.resize(frame, (680, 320))
#             (flag, encodedImage) = cv2.imencode(".jpg", frame)
#             if not flag:
#                 continue
#             yield (b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

#     def get_stream(self, freq: int = default_frame_frequency, status_code: int = 206,
#                 headers: Union[Mapping[str, str], None] = None,
#                 background: Union[BackgroundTasks, None] = None) -> StreamingResponse:

#         return StreamingResponse(self._start_stream(freq),
#                                 media_type="multipart/x-mixed-replace;boundary=frame",
#                                 status_code=status_code,
#                                 headers=headers,
#                                 background=background)



class FrameStreamer:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.vs = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()

    def _capture_frames(self):
        try:
            self.vs = cv2.VideoCapture(self.rtsp_url)
            self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if not self.vs.isOpened():
                raise RuntimeError(f"Failed to open video stream: {self.rtsp_url}")

            while True:
                ret, frame = self.vs.read()
                if not ret or frame is None:
                    continue
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
        except Exception as e:
            print(f"Error in frame capture: {e}")

    def _start_stream(self, freq: int):
        sleep_duration = 1.0 / freq
        while True:
            time.sleep(sleep_duration)
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                frame = cv2.resize(frame, (680, 320))
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if flag:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpg\r\n\r\n' +
                           bytearray(encodedImage) + b'\r\n')
                else:
                    continue

    def get_stream(self, freq: int = 10, status_code: int = 206,
                   headers: Union[Mapping[str, str], None] = None,
                   background: Union[BackgroundTasks, None] = None) -> StreamingResponse:

        return StreamingResponse(self._start_stream(freq),
                                 media_type="multipart/x-mixed-replace;boundary=frame",
                                 status_code=status_code,
                                 headers=headers,
                                 background=background)

@router.get("/video_feed/{site_id}")
async def video_feed(site_id: int, session: Session = Depends(utils.get_session)):
    db_site = session.get(Site, site_id)
    if not db_site:
        raise HTTPException(status_code=404, detail="Site not found")
    print(db_site.in_url)
    streamer = FrameStreamer(db_site.in_url)  # Use the actual URL from the database

    return streamer.get_stream()