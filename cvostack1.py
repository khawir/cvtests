from threading import Thread
import cv2, time
import os
import queue

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
 
class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        # Start the thread to read frames from the video stream
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.frame_queue = queue.Queue(maxsize=10)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_h = 720
        self.ratio = self.new_h/ self.height
        self.new_w = int(self.width * self.ratio)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    continue
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
            time.sleep(1.0/20)
    
    def show_frame(self):
        if not self.frame_queue.empty():
            self.frame = self.frame_queue.get()
        
        r_frame = cv2.resize(self.frame, (self.new_w, self.new_h) )
        cv2.imshow('frame', r_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

src = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716039080&id=711677256444084224&c=3cffb6de2e&t=e1498a314974763a69bcfa322cfc66b560034821f1069a0ad4701eb74382c53c&ev=100'

video_stream_widget = VideoStreamWidget(src)
while True:
    try:
        video_stream_widget.show_frame()
    except AttributeError:
        pass