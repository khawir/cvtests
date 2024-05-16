# import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

src = '3.mp4'

results = model.track(source=src,
                    #   conf=0.25,
                    #   iou=0.7,
                    #   imgsz=640,
                    #   half=False,
                      device='cuda:0',
                      max_det=10,
                      vid_stride=3,
                    #   stream_buffer=False,
                    #   visualize=False,
                      classes=[0],                      
                      show=True,
                      save_crop=True,
                      show_labels=True,
                      show_conf=False,
                      show_boxes=True,
                      line_width=1
                      )