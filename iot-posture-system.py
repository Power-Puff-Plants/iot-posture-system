# Import necessary libraries
import os
import time
import cv2
import numpy as np
import serial
import mediapipe as mp
from playsound import playsound
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Establish connection with Arduino via Raspberry Pi
try:
    arduino = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)  # Ensure the port is correct
    print("Arduino connection established")
except serial.SerialException:
    print("Unable to connect to Arduino")
    exit()
time.sleep(3)  # Wait for the connection to stabilize

# Initialize Firebase with credentials
cred = credentials.Certificate("../Keys/posture-plant-pal-firebase-adminsdk-1b6et-1a63e5a825.json")  # Update with your credentials file path
firebase_admin.initialize_app(cred)

# Set up Firestore database
db = firestore.client()

# Set up MediaPipe Pose and webcam feed
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)  # Open the default camera

# Global variables
countdown_start_time = None
calibration_start_time = None
good_posture_shoulder_y_threshold = 0
is_calibrated = False
alert_cooldown = 10  # Cooldown in seconds before the next alert
last_alert_time = 0
sound_file = "alert.mp3"  # Path to alert sound file
countdown_duration = 5  # Countdown duration in seconds
calibration_duration = 5  # Calibration duration in seconds
last_send_time = 0  # To track the last time data was sent to Firebase
last_serial_send_time = 0  # To track the last time data was sent to Arduino

# Function to send data to Firebase
def send_data_to_firebase(posture_status, shoulder_y_position):
    data = {
        'isPostureGood': posture_status == "Good Posture",
        'timeRecorded': datetime.utcnow()
    }
    # Sending data to Firestore
    db.collection('PostureData').add(data)  
    print("Data sent to Firebase:", data)

# Continuously capture frames from the webcam
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue  # Skip the loop iteration if frame reading fails

    # Convert the frame to RGB format for MediaPipe processing
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    current_timestamp = time.time()

    # Evaluate posture and trigger alerts if necessary
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))

        # Calculate the average Y position of both shoulders
        avg_shoulder_y = np.mean([left_shoulder[1], right_shoulder[1]])
        if avg_shoulder_y - 10 > good_posture_shoulder_y_threshold:
            posture_status = "Bad Posture"
            alert_color = (0, 0, 255)  # Red color for bad posture
            if current_timestamp - last_alert_time > alert_cooldown:
                print("Bad posture detected! Please correct your position.")
                if os.path.exists(sound_file):
                    playsound(sound_file)
                last_alert_time = current_timestamp
        else:
            posture_status = "Good Posture"
            alert_color = (0, 255, 0)  # Green color for good posture

        print(posture_status)

        # Send data to Firebase every 10 seconds
        if current_timestamp - last_send_time >= 10:
            send_data_to_firebase(posture_status, avg_shoulder_y)
            last_send_time = current_timestamp

        # Send serial data to Arduino every 5 seconds
        if current_timestamp - last_serial_send_time >= 5:
            if posture_status == "Good Posture":
                arduino.write(b'G')  # Send 'G' for good posture
            else:
                arduino.write(b'B')  # Send 'B' for bad posture
            last_serial_send_time = current_timestamp

# closing all necessary programs and exiting the script
cap.release()
cv2.destroyAllWindows()
arduino.close()