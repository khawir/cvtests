import threading
from fastapi import Depends, APIRouter, HTTPException
from sqlmodel import Session
from models import Site
from core import utils
import cv2
import time
import queue

#
from typing import Mapping, Union
import cv2
import time
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
#

router = APIRouter()
default_frame_frequency = 15

# class FrameStreamer:
#     def __init__(self, rtsp_url: str):
#         self.rtsp_url = rtsp_url
#         self.vs = None
#         self.frame_queue = queue.Queue(maxsize=10)
#         self.thread = threading.Thread(target=self._capture_frames)
#         self.thread.daemon = True
#         self.thread.start()

#     def _capture_frames(self):
#         try:
#             self.vs = cv2.VideoCapture(self.rtsp_url)
#             self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 2)
#             if not self.vs.isOpened():
#                 raise RuntimeError(f"Failed to open video stream: {self.rtsp_url}")

#             while True:
#                 ret, frame = self.vs.read()
#                 if not ret or frame is None:
#                     continue
#                 if self.frame_queue.full():
#                     self.frame_queue.get_nowait()
#                 self.frame_queue.put(frame)
#         except Exception as e:
#             print(f"Error in frame capture: {e}")

#     def _start_stream(self, freq: int):
#         sleep_duration = 1.0 / freq

#         while True:
#             time.sleep(sleep_duration)
#             if not self.frame_queue.empty():
#                 frame = self.frame_queue.get()
#                 frame = cv2.resize(frame, (680, 320))
#                 (flag, encodedImage) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
#                 if flag:
#                     yield (b'--frame\r\n'
#                     b'Content-Type: image/jpg\r\n\r\n' +
#                     bytearray(encodedImage) + b'\r\n')
#                 else:
#                     continue

#     def get_stream(self, freq: int = 10, status_code: int = 206,
#                 headers: Union[Mapping[str, str], None] = None,
#                 background: Union[BackgroundTasks, None] = None) -> StreamingResponse:

#         return StreamingResponse(self._start_stream(freq),
#                                 media_type="multipart/x-mixed-replace;boundary=frame",
#                                 status_code=status_code,
#                                 headers=headers,
#                                 background=background)

# @router.get("/video_feed/{site_id}")
# async def video_feed(site_id: int, session: Session = Depends(utils.get_session)):
#     # db_site = session.get(Site, site_id)
#     # if not db_site:
#     #     raise HTTPException(status_code=404, detail="Site not found")
#     # print(db_site.in_url)
#     static_url = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716049844&id=711722405662494720&c=3cffb6de2e&t=331a2b302bdbcba510b8c0cbc9f0b4138b7c2384baf5af3988a24fdc9f635961&ev=100"
#     streamer = FrameStreamer(static_url)

#     return streamer.get_stream()

class FrameStreamer:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.vs = None
        self.frame_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_frames(self):
        try:
            self.vs = cv2.VideoCapture(self.rtsp_url)
            self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if not self.vs.isOpened():
                raise RuntimeError(f"Failed to open video stream: {self.rtsp_url}")

            while not self.stop_event.is_set():
                ret, frame = self.vs.read()
                if not ret or frame is None:
                    continue
                
                # Wait for a free spot in the queue before adding the frame
                self.frame_queue.put(frame, block=True)
        except Exception as e:
            print(f"Error in frame capture: {e}")
        finally:
            if self.vs:
                self.vs.release()

    def _start_stream(self, freq: int):
        sleep_duration = 1.0 / freq
        try:
            while not self.stop_event.is_set():
                time.sleep(sleep_duration)
                try:
                    # Get the frame from the queue
                    frame = self.frame_queue.get(block=False)
                    frame = cv2.resize(frame, (680, 320))
                    (flag, encodedImage) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                    if flag:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpg\r\n\r\n' +
                               bytearray(encodedImage) + b'\r\n')
                    self.frame_queue.task_done()  # Mark the frame as processed
                except queue.Empty:
                    continue
        except GeneratorExit:
            self.stop()
        except Exception as e:
            print(f"Error in frame streaming: {e}")

    def stop(self):
        self.stop_event.set()
        self.capture_thread.join()

    def get_stream(self, freq: int = 10, status_code: int = 206,
                   headers: Union[Mapping[str, str], None] = None,
                   background: Union[BackgroundTasks, None] = None) -> StreamingResponse:
        if background:
            background.add_task(self.stop)
        return StreamingResponse(self._start_stream(freq),
                                 media_type="multipart/x-mixed-replace;boundary=frame",
                                 status_code=status_code,
                                 headers=headers)

@router.get("/video_feed/{site_id}")
async def video_feed(site_id: int, background: BackgroundTasks, session: Session = Depends(utils.get_session)):
    # db_site = session.get(Site, site_id)
    # if not db_site:
    #     raise HTTPException(status_code=404, detail="Site not found")
    # print(db_site.in_url)
    static_url = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716049844&id=711722405662494720&c=3cffb6de2e&t=331a2b302bdbcba510b8c0cbc9f0b4138b7c2384baf5af3988a24fdc9f635961&ev=100"
    streamer = FrameStreamer(static_url)
    return streamer.get_stream(background=background)