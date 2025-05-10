import cv2
import mediapipe as mp


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# All pose landmarks names (33 key points)
LANDMARK_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee", 
    "Right Knee", "Left Ankle", "Right Ankle", "Left Heel", "Right Heel", 
    "Left Foot Index", "Right Foot Index", "Left Toe", "Right Toe", 
    "Left Pinky", "Right Pinky", "Left Index", "Right Index", 
    "Left Thumb", "Right Thumb", "Left Inner Eye", "Right Inner Eye", 
    "Left Outer Eye", "Right Outer Eye"
]


def validate_human_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image_rgb)

    # Checking all body landmarks
    missing_landmarks = []
    for i, landmark in enumerate(results.pose_landmarks.landmark):
        if landmark.visibility < 0.5:  # visibility < 0.5 means the point is not visible
            missing_landmarks.append(LANDMARK_NAMES[i])
    
    return missing_landmarks
    