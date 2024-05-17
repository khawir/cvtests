import cv2
import queue
import threading
import time

class FrameStreamer:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.vs = None
        self.frame_queue = queue.Queue(maxsize=30)
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

    def start_display(self, freq: int):
        sleep_duration = 1.0 / freq
        try:
            while not self.stop_event.is_set():
                time.sleep(sleep_duration)

                try:
                    # Get the frame from the queue
                    frame = self.frame_queue.get(block=False)
                    frame = cv2.resize(frame, (680, 320))
                    cv2.imshow('Video Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()
                        break
                    self.frame_queue.task_done()  # Mark the frame as processed
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error in frame display: {e}")
        finally:
            cv2.destroyAllWindows()

    def stop(self):
        self.stop_event.set()
        self.capture_thread.join()

if __name__ == "__main__":
    static_url = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716063492&id=711779649986244608&c=3cffb6de2e&t=e4d1b2647a5f79e1d00724ab12ab399018f6a93caefce81d7eb213db71e68948&ev=100"
    streamer = FrameStreamer(static_url)
    streamer.start_display(freq=30)