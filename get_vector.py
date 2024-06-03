from deepface import DeepFace

img = "arif.png"

pers_vs = DeepFace.represent(
    img,
    model_name='SFace',
    enforce_detection=False,
    detector_backend='yolov8',
    # expand_percentage=10,
)

print(pers_vs)