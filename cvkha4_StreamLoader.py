class LoadStreams:

    def __init__(self, sources="file.streams", vid_stride=1, buffer=False):
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n  # frames per second
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # video capture objects
        self.imgs = [[] for _ in range(n)]  # images
        self.shape = [[] for _ in range(n)]  # image shapes
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            if urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                    "Try running 'source=0' in a local environment."
                )
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")  # newline

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")
        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns source paths, transformed and original images for processing."""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            # Wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  # q to quit
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"WARNING ⚠️ Waiting for stream {i}")

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self):
        """Return the length of the sources object."""
        return self.bs  # 1E12 frames = 32 streams at 30 FPS for 30 years
    


src = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716316220&id=712839666379337728&c=3cffb6de2e&t=c75230bb20b920e2dfebae890457b6528ae9a2f693d8e83f62500787b2057ce7&ev=100"

loaders_obj = LoadStreams(src)

for data in loaders_obj:
    image_frame = data[1]  # Assuming data is the list you provided
    try:
        cv2.imshow("Image", image_frame)
        cv2.waitKey(1)  # Wait for 1 millisecond for key press
        cv2.destroyAllWindows()  # Close the window when a key is pressed
    except Exception as e:
        print(f"Error displaying frame: {e}")