import cv2
import threading
import queue
import time

class RTSPStream:
    def __init__(self, rtsp_url, frame_rate=30, buffer_size=30):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.frame_rate = frame_rate
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.running = True

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_h = 720
        self.ratio = self.new_h/ self.height
        self.new_w = int(self.width * self.ratio)

        # Start the frame capturing thread
        self.thread = threading.Thread(target=self.update_frame, args=())
        self.thread.daemon = True
        self.thread.start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.queue.full():
                    self.queue.put(frame)
                else:
                    # Drop the oldest frame to make space for the new one
                    try:
                        self.queue.get_nowait()
                        self.queue.put(frame)
                    except queue.Empty:
                        pass
            else:
                time.sleep(0.01)  # Sleep briefly if the read fails to avoid busy-waiting

    def get_frame(self):
        try:
            self.frame = self.queue.get_nowait()
            return cv2.resize(self.frame, (self.new_w, self.new_h) )
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# Usage
rtsp_url = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716039080&id=711677256444084224&c=3cffb6de2e&t=e1498a314974763a69bcfa322cfc66b560034821f1069a0ad4701eb74382c53c&ev=100"
frame_rate = 20  # Set the desired frame rate
stream = RTSPStream(rtsp_url, frame_rate=frame_rate)

while True:
    start_time = time.time()
    frame = stream.get_frame()
    if frame is not None:
        cv2.imshow('RTSP Stream', frame)

    # Maintain the desired frame rate
    elapsed_time = time.time() - start_time
    time_to_wait = max(1. / frame_rate - elapsed_time, 0)
    if cv2.waitKey(int(time_to_wait * 1000)) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
