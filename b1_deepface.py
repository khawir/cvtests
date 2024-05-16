from deepface import DeepFace
import numpy as np
import cv2
import json
import time


models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

backends = [
  'opencv', 
  'ssd', 
  # 'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

metrics = ["cosine", "euclidean", "euclidean_l2"]

actions = ['age', 'gender', 'race', 'emotion']



# ---- Face Detection ----
# ------------------------

# img = "oscar.jpg"
# image = cv2.imread(img)

# face_objs = DeepFace.extract_faces(
#   img_path = img, 
#   detector_backend = backends[7],
# )

# for i, face_obj in enumerate(face_objs):
#     x, y, w, h, _, _ = face_obj["facial_area"].values()    
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)



# cv2.imshow('frame', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# -------- Stream --------
# ------------------------

# src = '1.mp4'

# DeepFace.stream("db",
#                 model_name=models[0],
#                 detector_backend = backends[7],
#                 distance_metric= metrics[0],
#                 enable_face_analysis=True,
#                 source=src,
#                 time_threshold=10,
#                 frame_threshold=10
#                 )



# ----- Demographies -----
# ------------------------

# for backend in backends:
#   tic = time.time()
#   results = DeepFace.analyze(
#     img_path = "emilia.jpg", 
#     actions = ['gender'],
#     detector_backend = backend,
#     # expand_percentage = 10,
#   )
#   toc = time.time()
#   print(json.dumps(results))
#   # for result in results:
#   #   print(result['dominant_gender'])
#   print(f"{backend}: {toc-tic} seconds")

# # for i, demography in enumerate(demographies):
# #     print(demography)



# ----- Backend Speed ----
# ------------------------


for backend in backends:
  tic = time.time()
  results = DeepFace.extract_faces(
    img_path = "emilia.jpg", 
    detector_backend = backend,
  )
  toc = time.time()
  # print(json.dumps(results))
  for result in results:
    print(result['facial_area'])
  print(f"{backend}: {toc-tic} seconds")