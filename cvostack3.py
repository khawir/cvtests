import cv2
import threading
import queue
import time

class RTSPStreamViewer:
    def __init__(self, rtsp_url, frame_rate=30, queue_size=10):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.frame_rate = frame_rate
        self.buffer_size = queue_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.start()
        self.frame_generator_thread = threading.Thread(target=self.frame_generator_loop)
        self.frame_generator_thread.start()

    def display_loop(self):
        desired_time_between_frames = 1.0 / self.frame_rate
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.1)  # Non-blocking queue access
                if frame is not None:
                    cv2.imshow('RTSP Stream', frame)
                if cv2.waitKey(1) == 27:  # Press Esc to quit
                    break
            except queue.Empty:
                pass  # Handle empty queue gracefully

            elapsed_time = time.time() - start_time
            time_to_wait = max(desired_time_between_frames - elapsed_time, 0)
            cv2.waitKey(int(time_to_wait * 1000))  # Wait for desired frame rate

    def frame_generator_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Drop the oldest frame (or explore alternative dropping strategy)
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(frame)
            else:
                break

    def run(self):
        self.display_thread.join()
        self.frame_generator_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rtsp_url = 'https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716039080&id=711677256444084224&c=3cffb6de2e&t=e1498a314974763a69bcfa322cfc66b560034821f1069a0ad4701eb74382c53c&ev=100https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716039080&id=711677256444084224&c=3cffb6de2e&t=e1498a314974763a69bcfa322cfc66b560034821f1069a0ad4701eb74382c53c&ev=100'  #
