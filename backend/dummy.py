import base64
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit # type: ignore
import statistics
import time
import math 
font = cv2.FONT_HERSHEY_SIMPLEX 

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


def get_unique(c):
    temp_list = list(c)
    temp_set = set()
    for t in temp_list:
        temp_set.add(t[0])
        temp_set.add(t[1])
    return list(temp_set)

mp_face_mesh = mp.solutions.face_mesh
connections_iris = mp_face_mesh.FACEMESH_IRISES
iris_indices = get_unique(connections_iris)

iris_right_horzn = [469,471]
iris_right_vert = [470,472]
iris_left_horzn = [474,476]
iris_left_vert = [475,477]


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(distance)


def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate the centroid of a set of points
def calculate_centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return np.array([centroid_x, centroid_y])

# Function to calculate the area of a polygon defined by points
def calculate_polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def live_distance(frame,results):
    for face_landmark in results.multi_face_landmarks:
        lms = face_landmark.landmark
        d= {}
        for index in iris_indices:
            x = int(lms[index].x*frame.shape[1])
            y = int(lms[index].y*frame.shape[0])
            d[index] = (x,y)
    
        
        
        centre_right_iris_x_1 = int((d[iris_right_horzn[0]][0] + d[iris_right_horzn[1]][0])/2)
        centre_right_iris_y_1 = int((d[iris_right_horzn[0]][1] + d[iris_right_horzn[1]][1])/2)
        
        centre_right_iris_x_2 = int((d[iris_right_vert[0]][0] + d[iris_right_vert[1]][0])/2)
        centre_right_iris_y_2 = int((d[iris_right_vert[0]][1] + d[iris_right_vert[1]][1])/2)
        
            
        centre_left_iris_x_1 = int((d[iris_left_horzn[0]][0] + d[iris_left_horzn[1]][0])/2)
        centre_left_iris_y_1 = int((d[iris_left_horzn[0]][1] + d[iris_left_horzn[1]][1])/2)
        
        centre_left_iris_x_2 = int((d[iris_left_vert[0]][0] + d[iris_left_vert[1]][0])/2)
        centre_left_iris_y_2 = int((d[iris_left_vert[0]][1] + d[iris_left_vert[1]][1])/2)
        
        centre_left_iris_x = int((centre_left_iris_x_1 + centre_left_iris_x_2)/2)
        centre_left_iris_y = int((centre_left_iris_y_1 + centre_left_iris_y_2)/2)
        
        centre_right_iris_x = int((centre_right_iris_x_1 + centre_right_iris_x_2)/2)
        centre_right_iris_y = int((centre_right_iris_y_1 + centre_right_iris_y_2)/2)
        
        cv2.circle(frame,(centre_right_iris_x,centre_right_iris_y),2,(0,255,0),-1)
        cv2.circle(frame,(centre_left_iris_x,centre_left_iris_y),2,(0,255,0),-1)
        
        w = ((centre_right_iris_x - centre_left_iris_x)**2 + (centre_right_iris_y - centre_left_iris_y)**2)**0.5
        
        W = 6.3
        
        f = 1024
        
        d = f*W/w
    return d


def angle_between_three_points_normalized(p1, p2, p3, AL):
    # Normalize the points based on the axial length
    p1_norm = np.array(p1) / AL
    p2_norm = np.array(p2) / AL
    p3_norm = np.array(p3) / AL
    
    # Create vectors from normalized points
    vec_p1p2_norm = p2_norm - p1_norm
    vec_p2p3_norm = p3_norm - p2_norm
    
    # Calculate the dot product between normalized vectors
    dot_product = np.dot(vec_p1p2_norm, vec_p2p3_norm)
    
    # Calculate the magnitudes of normalized vectors
    magnitude_p1p2_norm = np.linalg.norm(vec_p1p2_norm)
    magnitude_p2p3_norm = np.linalg.norm(vec_p2p3_norm)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_p1p2_norm * magnitude_p2p3_norm)
    
    # Calculate the angle in radians and then convert it to degrees
    theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_degrees = np.degrees(theta_radians)
    
    return theta_degrees

def calculate_head_orientation(landmarks):
    # Assume landmarks are normalized [0, 1]l
    # Define some key landmarks
    nose_tip = landmarks[1]  # Tip of the nose
    nose_bridge = landmarks[6]  # Top of the nose bridge
    left_eye_outer = landmarks[33]  # Outer corner of the left eye
    right_eye_outer = landmarks[263]  # Outer corner of the right eye
    # Convert landmarks to numpy arrays
    nose_tip = np.array([nose_tip.x, nose_tip.y, nose_tip.z])
    nose_bridge = np.array([nose_bridge.x, nose_bridge.y, nose_bridge.z])
    left_eye_outer = np.array([left_eye_outer.x, left_eye_outer.y, left_eye_outer.z])
    right_eye_outer = np.array([right_eye_outer.x, right_eye_outer.y, right_eye_outer.z])
    # Calculate the vectors
    horizontal_vector = right_eye_outer - left_eye_outer
    vertical_vector = nose_bridge - nose_tip
    # Normalize the vectors
    horizontal_vector_normalized = horizontal_vector / np.linalg.norm(horizontal_vector)
    vertical_vector_normalized = vertical_vector / np.linalg.norm(vertical_vector)
    # Calculate roll
    roll = np.arctan2(horizontal_vector_normalized[1], horizontal_vector_normalized[0])
    roll = np.degrees(roll)
    # Calculate yaw and pitch
    # This is a simplified approach - for more accuracy, a 3D head model or additional landmarks might be necessary
    yaw = np.arctan2(vertical_vector_normalized[0], vertical_vector_normalized[2])
    yaw = np.degrees(yaw)
    pitch = np.arctan2(vertical_vector_normalized[1], vertical_vector_normalized[2])
    pitch = np.degrees(pitch)
    return roll, yaw, pitch

def draw_specific_landmarks(frame, landmarks, indices):
    for connection in indices:
        if len(connection) == 2:
            start_idx, end_idx = connection
            if 0 <= start_idx < len(landmarks.landmark) and 0 <= end_idx < len(landmarks.landmark):
                start_landmark = landmarks.landmark[start_idx]
                end_landmark = landmarks.landmark[end_idx]
                cv2.line(frame, (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])),
                               (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])), (0, 255, 0), 1)
def draw_eye_bounding_box(frame, landmarks, indices):
    min_x, min_y = frame.shape[1], frame.shape[0]
    max_x, max_y = 0, 0

    for connection in indices:
        start_idx, end_idx = connection
        for idx in [start_idx, end_idx]:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)


def get_face_roi(landmarks, image):
    """
    Determine the region of interest of the face based on landmarks.
    """
    # Get the bounding box coordinates
    x_coordinates = [int(landmark.x * image.shape[1]) for landmark in landmarks]
    y_coordinates = [int(landmark.y * image.shape[0]) for landmark in landmarks]
    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)
    return x_min, y_min, x_max, y_max

def calculate_rotation_angle(landmarks, image):
    """
    Calculate the rotation angle of the face based on eye landmarks.
    """
    # Define eye landmarks (indices may vary based on MediaPipe's output format)
    left_eye = landmarks[33]  # Example index for left eye
    right_eye = landmarks[263] # Example index for right eye
    # Calculate angle
    eye_line = [int(right_eye.x * image.shape[1]) - int(left_eye.x * image.shape[1]),
                int(right_eye.y * image.shape[0]) - int(left_eye.y * image.shape[0])]
    angle = math.atan2(eye_line[1], eye_line[0])
    return math.degrees(angle)
def rotate_image(image, angle, center=None, scale=1.0):
    """
    Rotate the image by a given angle.
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
def perspective_transform(image, x_min, y_min, x_max, y_max, rotation_angle):
    """
    Apply perspective transform and rotation to the face region.
    """
    # Rotate the image first
    rotated_image = rotate_image(image, rotation_angle)
    # Updated points for perspective transform
    points1 = np.float32([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
    points2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    return cv2.warpPerspective(rotated_image, matrix, (500, 500))


LEFT_EYE_INDICES = mp_face_mesh.FACEMESH_LEFT_EYE
RIGHT_EYE_INDICES = mp_face_mesh.FACEMESH_RIGHT_EYE
LEFT_IRIS_INDICES = mp_face_mesh.FACEMESH_LEFT_IRIS
LEFT_IRIS_INDICES = get_unique(LEFT_IRIS_INDICES)
RIGHT_IRIS_INDICES = mp_face_mesh.FACEMESH_RIGHT_IRIS
RIGHT_IRIS_INDICES = get_unique(RIGHT_IRIS_INDICES)
imp_indexes = LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES
eyes_indices = [130, 133, 359, 362]


mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    
mp_face_mesh = mp.solutions.face_mesh
connections_iris = mp_face_mesh.FACEMESH_IRISES
iris_indices = get_unique(connections_iris)


pos_similarities = []

for i in eyes_indices:
    imp_indexes.append(i)

font = cv2.FONT_HERSHEY_SIMPLEX  # You can choose different fonts
position = (50, 50)  # Coordinates of the bottom-left corner of the text string in the image
font_scale = 1  # Font scale factor
color = (255, 0, 0)  # Color in BGR (not RGB, be careful about this)
thickness = 1  # Thickness of the lines used to draw the text



@socketio.on('connect', namespace='/hello')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect', namespace='/hello')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('video_frame', namespace='/hello')
def handle_video_frame(image_data):
    print("Received video frame data")
    print("Image data:", image_data)
    # Decode Base64-encoded image data
    face_detection = mp_face_detection.FaceDetection()
    face_mesh = mp_face_mesh.FaceMesh()
    image_data_decoded = base64.b64decode(image_data.split(',')[1])

    # Convert decoded data to NumPy array
    nparr = np.frombuffer(image_data_decoded, np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    try:
        if results.detections:
            for detection in results.detections:
                # Use face mesh to get landmarks
                mesh_results = face_mesh.process(image_rgb)
                if mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        
                        dist = live_distance(image,mesh_results)
                        dist = round(dist,2)
                        cv2.putText(image,"Distance : " + str(dist), (50,300), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
                        #print(dist)
                        

                        rotation_angle = calculate_rotation_angle(face_landmarks.landmark, image)
                        x_min, y_min, x_max, y_max = get_face_roi(face_landmarks.landmark, image)
                        transformed_face = perspective_transform(image, x_min, y_min, x_max, y_max, rotation_angle)
                        roll, yaw, pitch = calculate_head_orientation(face_landmarks.landmark)
                        image = cv2.putText(image, f'Roll: {roll:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        image = cv2.putText(image, f'Yaw: {yaw:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        image = cv2.putText(image, f'Pitch: {pitch:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                        transformed_face_rgb = cv2.cvtColor(transformed_face, cv2.COLOR_BGR2RGB)
                        results_transformed = face_mesh.process(transformed_face_rgb)

                        LEFT_EYE_INDICES = [362,385,387,263,373,380]
                        RIGHT_EYE_INDICES = [33,160,158,133,153,144]
                        LEFT_IRIS_INDICES = [474, 475, 476, 477]
                        RIGHT_IRIS_INDICES = [469, 470, 471, 472]
                        
                        left_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]) for i in LEFT_EYE_INDICES])
                        right_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]) for i in RIGHT_EYE_INDICES])
                        left_iris_points = np.array([(face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]) for i in LEFT_IRIS_INDICES])
                        right_iris_points = np.array([(face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]) for i in RIGHT_IRIS_INDICES])

                        left_ear = calculate_ear(left_eye_points)
                        right_ear = calculate_ear(right_eye_points)
                        left_eye_area = calculate_polygon_area(left_eye_points)
                        right_eye_area = calculate_polygon_area(right_eye_points)
                        left_iris_centroid = calculate_centroid(left_iris_points)
                        right_iris_centroid = calculate_centroid(right_iris_points)
                        pupillary_distance = np.linalg.norm(left_iris_centroid - right_iris_centroid)
            
                        # Display metrics
                        cv2.putText(image, f'Left EAR: {left_ear:.2f}', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(image, f'Right EAR: {right_ear:.2f}', (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(image, f'Left Eye Area: {left_eye_area:.2f}', (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(image, f'Right Eye Area: {right_eye_area:.2f}', (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(image, f'Pupillary Distance: {pupillary_distance:.2f}px', (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        if results_transformed.multi_face_landmarks:
                            for face_landmarks in results_transformed.multi_face_landmarks:
                                # draw_eye_bounding_box(transformed_face, face_landmarks, LEFT_EYE_INDICES)
                                # draw_eye_bounding_box(transformed_face, face_landmarks, RIGHT_EYE_INDICES)

                                left_x = 0
                                left_y = 0
                                right_x = 0
                                right_y = 0
                                
                                for index in imp_indexes:
                                    if index < len(face_landmarks.landmark):
                                        iris_landmark = face_landmarks.landmark[index]
                                        if index == 469 or index == 470 or index == 471 or index == 472:
                                            x = int(iris_landmark.x * transformed_face.shape[1])
                                            y = int(iris_landmark.y * transformed_face.shape[0])
                                            right_x+= x 
                                            right_y+= y 
                                            cv2.circle(transformed_face, (x, y), 2, (0, 255, 0), -1)
                    
                                        elif index == 474 or index == 475 or index == 476 or index == 477:  
                                            x = int(iris_landmark.x * transformed_face.shape[1])
                                            y = int(iris_landmark.y * transformed_face.shape[0])
                                            left_x+= x 
                                            left_y+= y 
                                            cv2.circle(transformed_face, (x, y), 2, (0, 255, 0), -1)
                                            
                                        else:
                                            
                                            x = int(iris_landmark.x * transformed_face.shape[1])
                                            y = int(iris_landmark.y * transformed_face.shape[0])
                                            cv2.circle(transformed_face, (x, y), 2, (0, 255, 0), -1)
                                            
                                            
                                right_x = int(right_x/4)
                                right_y = int(right_y/4)
                                left_x = int(left_x/4)
                                left_y = int(left_y/4)
                                cv2.circle(transformed_face, (right_x, right_y), 2, (0, 255, 0), -1)
                                # cv2.circle(transformed_face, (left_x, left_y), 2, (0, 255, 0), -1)
                                left_eye_left_point_index = 359
                                left_eye_right_point_index = 362
                                right_eye_right_point_index = 130
                                right_eye_left_point_index = 133
                                
                                le_lp_x = int(face_landmarks.landmark[359].x * transformed_face.shape[1])
                                le_lp_y = int(face_landmarks.landmark[359].y * transformed_face.shape[0])
                                le_rp_x = int(face_landmarks.landmark[362].x * transformed_face.shape[1])
                                le_rp_y = int(face_landmarks.landmark[362].y * transformed_face.shape[0])

                                re_lp_x = int(face_landmarks.landmark[133].x * transformed_face.shape[1])
                                re_lp_y = int(face_landmarks.landmark[133].y * transformed_face.shape[0])
                                re_rp_x = int(face_landmarks.landmark[130].x * transformed_face.shape[1])
                                re_rp_y = int(face_landmarks.landmark[130].y * transformed_face.shape[0])

                                axial_left_eye = calculate_distance(le_lp_x,le_lp_y,le_rp_x,le_rp_y)
                                axial_right_eye = calculate_distance(re_lp_x,re_lp_y,re_rp_x,re_rp_y)
                                
                                R1_point = (re_lp_x,re_lp_y)
                                R2_point = (re_rp_x,re_rp_y)
                                L2_point = (le_lp_x,le_lp_y)
                                L1_point = (le_rp_x,le_rp_y)
                                
                                right_eye_angle = angle_between_three_points_normalized(R1_point,(right_x,right_y),R2_point,axial_right_eye)
                                left_eye_angle = angle_between_three_points_normalized(L2_point,(left_x,left_y),L1_point,axial_left_eye)
                                angle_difference = right_eye_angle - left_eye_angle
                                angle_difference = round(angle_difference,2)
                                #print(axial_left_eye)
                                
                                                    
                                L1 = 0 
                                L2 = 0 
                                R1 = 0
                                R2 = 0
                                for index in eyes_indices:
                                    if index < len(face_landmarks.landmark):
                                        iris_landmark = face_landmarks.landmark[index]
                                        if index == 130:
                                            x = int(iris_landmark.x * transformed_face.shape[1])
                                            y = int(iris_landmark.y * transformed_face.shape[0])
                                            R2 = calculate_distance(x,y,right_x,right_y)
                                        if index == 133:
                                            x = int(iris_landmark.x * transformed_face.shape[1])
                                            y = int(iris_landmark.y * transformed_face.shape[0])
                                            R1 = calculate_distance(x,y,right_x,right_y)
                                        if index == 359:
                                            x = int(iris_landmark.x * transformed_face.shape[1])
                                            y = int(iris_landmark.y * transformed_face.shape[0])
                                            L2 = calculate_distance(x,y,left_x,left_y)
                                        if index == 362:
                                            x = int(iris_landmark.x * transformed_face.shape[1])
                                            y = int(iris_landmark.y * transformed_face.shape[0])
                                            L1 = calculate_distance(x,y,left_x,left_y)

                                #dist = live_distance(mesh_results)
                                #print(dist)
                                # print("R1 : " + str(R1))
                                # print("R2 : " + str(R2))
                                # print("L1 : " + str(L1))
                                # print("L2 : " + str(L2))

                                k1= (R1/R2)/axial_right_eye
                                k2 = (L2/L1)/axial_left_eye
                                pos_sim = max((k1),(k2))/min((k1),(k2))
                                pos_sim = round(pos_sim,2)

                                try:
                                    pos_sim2 = k1/k2
                                    pos_sim2 = round(pos_sim2,2)
                                except Exception as e:
                                    print(e)
                                # pos_sim = abs(float(R2/R1) - float(L1/L2))
                                # pos_sim = normalized_pos_sim(R1,R2,L1,L2)
                                pos_similarities.append(pos_sim)

                                text = "Metric1 : " + str(pos_sim)

                                text2 = "Metric2 : " + str(pos_sim2)

                                pos_sim3 = abs(k2-k1)
                                pos_sim3 = round(pos_sim3,4)
                                text3 = "Metric3 : " + str(pos_sim3)
                                text4 = "Metric4 : " + str(angle_difference)
                                # Using cv2.putText() method
                                cv2.putText(transformed_face, text, (50,250), font, font_scale, color, thickness, cv2.LINE_AA)
                                cv2.putText(transformed_face,text2, (50,300), font, font_scale, color, thickness, cv2.LINE_AA)
                                cv2.putText(transformed_face,text3, (50,350), font, font_scale, color, thickness, cv2.LINE_AA)
                                cv2.putText(transformed_face,text4, (50,400), font, font_scale, color, thickness, cv2.LINE_AA) 

                                yield pos_sim, pos_sim2, pos_sim3, angle_difference     
                                
            # Display the frames
            # cv2.imshow('Original Frame', image)
            # print("yes boss")
            # cv2.imshow('Transformed Face', transformed_face)    
    except Exception as e:
        print(e, "error hai boss")    
    # Show the image
    cv2.imshow('MediaPipe Eye Tracking', image)
    print('MediaPipe Eye Tracking', image)
    emit('eye_detection_result', {'num_eyes_detected': '2'}, namespace='/hello')    
        
@app.route('/')
def index():
    return 'Hello, World!'


if __name__ == '__main__':
    socketio.run(app)
