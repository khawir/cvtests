import os
from deepface import DeepFace

folder_path = "cls/"

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, filename)

        pers_gs = DeepFace.analyze(
            image_path,
            actions = ['gender'],
            enforce_detection=False,
            detector_backend='yolov8',
            expand_percentage=10,
            # silent=True
            )
        
        print(f"{image_path} : {pers_gs[0]['dominant_gender']}")
