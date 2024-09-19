import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Global variables
countdown_start_time = None
calibration_start_time = None
good_posture_shoulder_y_threshold = 0
is_calibrated = False
alert_cooldown = 10  # Cooldown in seconds before next alert
last_alert_time = 0
sound_file = "alert.mp3"  # Path to alert sound file
countdown_duration = 5  # Countdown duration in seconds
calibration_duration = 5  # Calibration duration in seconds

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points (point1 -> point2 -> point3).
    """
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    current_time = time.time()

    if countdown_start_time is not None:
        # Countdown Phase
        elapsed_time = current_time - countdown_start_time
        remaining_time = countdown_duration - int(elapsed_time)
        if remaining_time > 0:
            cv2.putText(frame, f"Prepare for calibration in {remaining_time} seconds...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            countdown_start_time = None
            calibration_start_time = current_time
            print("Calibration started. Please set your good posture.")
    elif not is_calibrated:
        # Calibration Phase
        if calibration_start_time is None:
            calibration_start_time = current_time
            print("Calibrating for 10 seconds. Please set your good posture.")
        elapsed_time = current_time - calibration_start_time
        if elapsed_time < calibration_duration:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
                right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                                  int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))

                shoulder_y_positions = [left_shoulder[1], right_shoulder[1]]
                good_posture_shoulder_y_threshold = min(shoulder_y_positions)
                cv2.putText(frame, f"Calibrating... {int(elapsed_time)}/{calibration_duration} seconds", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            is_calibrated = True
            print(f"Calibration complete. Good posture threshold set at Y = {good_posture_shoulder_y_threshold}.")
            calibration_start_time = None
    else:

        # Posture Evaluation
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
            right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))

            shoulder_y_position = np.mean([left_shoulder[1], right_shoulder[1]])
            if shoulder_y_position - 75 > good_posture_shoulder_y_threshold:
                status = "Bad Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > alert_cooldown:
                    print("Bad posture detected! Please correct your position.")
                    if os.path.exists(sound_file):
                        playsound(sound_file)
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Y Position: {shoulder_y_position}/{good_posture_shoulder_y_threshold}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()