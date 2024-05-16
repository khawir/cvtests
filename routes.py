from fastapi import FastAPI, Request, APIRouter, HTTPException, WebSocket, Response
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import psycopg2
import asyncio
import base64
import json
import cv2
import time

router = APIRouter()
clients = set()

DATABASE_URL = 'postgresql://postgres:1234@localhost:5432/TestingDB'

BUFFER_SIZE = 10  # Number of frames to accumulate before sending
BUFFER_TIMEOUT = 5  # Time in seconds to wait before sending the buffer

connection = psycopg2.connect(DATABASE_URL)
connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
cursor = connection.cursor()
cursor.execute("LISTEN new_row;")
print("Listening for notifications on 'new_row' channel...")

s = "https://isgpopen.ezvizlife.com/v3/openlive/AA4823505_1_1.m3u8?expire=1715512589&id=709468992766025728&c=3cffb6de2e&t=c5d248d8aa83308d29d40463da6eac19cd6e220bf82eeff185bf2672b8e9ef41&ev=100"

# @router.get("/video_feed")
# async def video_feed():
#     source = cv2.VideoCapture(s)
#     try:
#         frame_count = 0
#         prev_frame_time = time.time()
#         target_fps = 15
#         frame_interval = 1 / target_fps

#         while True:
#             success, frame = source.read()
#             if not success:
#                 yield Response(content="No frames available", media_type="text/plain")
#                 continue
#             else:
#                 # Resize and compress the frame for optimization
#                 frame = cv2.resize(frame, (320, 240))  # Smaller resolution
#                 ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])  # Lower compression quality
#                 frame = buffer.tobytes()
#                 frame_base64 = base64.b64encode(frame).decode()
#                 yield f"data:image/jpeg;base64,{frame_base64}"

#                 curr_frame_time = time.time()
#                 delta_time = curr_frame_time - prev_frame_time

#                 if delta_time < frame_interval:
#                     continue

#                 prev_frame_time = curr_frame_time
#                 frame_count += 1
#     except Exception as e:
#         print(f"Error: {e}")
#         yield Response(content="Stream interrupted", media_type="text/plain")
#     finally:
#         source.release()

# def gen_frames():  
#     source = cv2.VideoCapture(s)

#     while True:
#         has_frame, frame = source.read()
#         if not has_frame:
#             print("sorry, no camera")
#             break
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @router.get("/video_feed")
# async def video_feed():
#     return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@router.get("/video_feed")
async def video_feed():
    return EventSourceResponse(generate_frames())

def generate_frames():
    video = cv2.VideoCapture(s)
    try:
        while True:
            success, frame = video.read()
            if not success:
                yield dict(event="NO_FRAMES", data="No frames available")
                continue
            else:
                frame = cv2.resize(frame, (640, 480))
                #Decrease the quality of frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])
                frame = buffer.tobytes()
                frame_base64 = base64.b64encode(frame).decode()
                yield f"data:image/jpeg;base64,{frame_base64}"

                cv2.waitKey(75)
    except Exception as e:
        print(f"Error: {e}")
        yield dict(event="STREAM_INTERRUPTED", data="Error Stream Interrupted")
    finally:
        video.release()

# @router.get("/stream")
# async def stream():
#     queue = asyncio.Queue()
#     clients.add(queue)

#     async def event_generator():
#         try:
#             while True:
#                 connection.poll() # Get the message
#                 while connection.notifies:
#                     notification = connection.notifies.pop() # Pop notification from list
#                     print(f"channel: {notification.channel}")
#                     print(f"message: {json.loads(notification.payload)}")
#                     for client in clients:
#                         print("sending data to connected clients...")
#                         await client.put(json.dumps({"data":notification.payload}))
#                     data = await queue.get()
#                     yield f"data: {data}\n\n"

#                 await asyncio.sleep(1)

#         except asyncio.CancelledError:          # This happens when the client disconnects
#             clients.remove(queue)
#             raise
#     # Create a streaming response using the event generator
#     return EventSourceResponse(event_generator())




