from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams
import cv2
import numpy as np

s1 = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1716316220&id=712839666379337728&c=3cffb6de2e&t=c75230bb20b920e2dfebae890457b6528ae9a2f693d8e83f62500787b2057ce7&ev=100"
s2 = 'rtsp://admin:hik@12345@172.23.16.55'

src = [s1, s2]

loaders_obj = LoadStreams(sources="file.streams")


for results in loaders_obj:
    for result in results:
        print(result[1])



# for data in loaders_obj:
#     image_frame = data[1]  # Assuming data is the list you provided

#     if isinstance(image_frame, np.ndarray) and image_frame.size > 0:
#     # Convert the frame data type if necessary (assuming uint8)
#     # image_frame = image_frame.astype(np.uint8)  # Uncomment if needed

#     # Display the frame with error handling
#         try:
#             cv2.imshow("Image", image_frame)
#             cv2.waitKey(1)  
#             cv2.destroyAllWindows()  # Close the window when a key is pressed
#         except Exception as e:
#             print(f"Error displaying frame: {e}")