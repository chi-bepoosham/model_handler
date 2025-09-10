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

# فارسی‌سازی نام نقاط کلیدی
LANDMARK_TRANSLATIONS = {
    "Nose": "بینی",
    "Left Eye": "چشم چپ",
    "Right Eye": "چشم راست",
    "Left Ear": "گوش چپ",
    "Right Ear": "گوش راست",
    "Left Shoulder": "شانه چپ",
    "Right Shoulder": "شانه راست",
    "Left Elbow": "آرنج چپ",
    "Right Elbow": "آرنج راست",
    "Left Wrist": "مچ دست چپ",
    "Right Wrist": "مچ دست راست",
    "Left Hip": "باسن چپ",
    "Right Hip": "باسن راست",
    "Left Knee": "زانو چپ",
    "Right Knee": "زانو راست",
    "Left Ankle": "قوزک پا چپ",
    "Right Ankle": "قوزک پا راست",
    "Left Heel": "پاشنه چپ",
    "Right Heel": "پاشنه راست",
    "Left Foot Index": "انگشت پای چپ (اشاره)",
    "Right Foot Index": "انگشت پای راست (اشاره)",
    "Left Toe": "انگشتان پای چپ",
    "Right Toe": "انگشتان پای راست",
    "Left Pinky": "انگشت کوچک دست چپ",
    "Right Pinky": "انگشت کوچک دست راست",
    "Left Index": "انگشت اشاره دست چپ",
    "Right Index": "انگشت اشاره دست راست",
    "Left Thumb": "انگشت شست چپ",
    "Right Thumb": "انگشت شست راست",
    "Left Inner Eye": "گوشه داخلی چشم چپ",
    "Right Inner Eye": "گوشه داخلی چشم راست",
    "Left Outer Eye": "گوشه بیرونی چشم چپ",
    "Right Outer Eye": "گوشه بیرونی چشم راست"
}

# Key landmarks necessary for full-body validation
REQUIRED_LANDMARKS = [
    "Nose", "Left Eye", "Right Eye",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist",
    "Left Hip", "Right Hip", "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle", "Left Heel", "Right Heel",
    "Left Foot Index", "Right Foot Index", "Left Toe", "Right Toe" 
]

def validate_human_image(image_path):
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    missing_landmarks = []
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            name = LANDMARK_NAMES[i]
            if name in REQUIRED_LANDMARKS and landmark.visibility < 0.5:
                missing_landmarks.append(LANDMARK_TRANSLATIONS.get(name, name))
    else:
        missing_landmarks = [LANDMARK_TRANSLATIONS.get(l, l) for l in REQUIRED_LANDMARKS]

    return missing_landmarks if missing_landmarks else []
