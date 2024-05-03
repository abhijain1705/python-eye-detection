import base64
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit # type: ignore

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='http://localhost:3000')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    
# Initialize MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

@socketio.on('video_frame')
def handle_video_frame(image_data):
    # Decode Base64-encoded image data
    image_data_decoded = base64.b64decode(image_data.split(',')[1])

    # Convert decoded data to NumPy array
    nparr = np.frombuffer(image_data_decoded, np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the image and find faces
    results = face_mesh.process(image)
    
    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh annotations on the image
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Specifically for the eyes
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Indices of left eye (around the eye)
                if idx in range(133, 154):
                    h, w, c = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
                # Indices of right eye (around the eye)
                if idx in range(362, 384):
                    h, w, c = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Show the image
    cv2.imshow('MediaPipe Eye Tracking', image)
    print('MediaPipe Eye Tracking', image)
    emit('eye_detection_result', {'num_eyes_detected': 2})

@app.route('/')
def index():
    return 'Hello, World!'


if __name__ == '__main__':
    socketio.run(app)
