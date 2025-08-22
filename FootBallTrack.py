# -------------------------------
# Install required packages
# -------------------------------
# !pip install ultralytics
# !pip install opencv-python
# !pip install mediapipe
# !pip install numpy
# !pip install matplotlib

# -------------------------------
# Import libraries
# -------------------------------
import torch
import ultralytics
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from google.colab.patches import cv2_imshow

# -------------------------------
# Print library versions
# -------------------------------
print("Torch:", torch.__version__)
print("YOLOv8:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("Mediapipe:", mp.__version__)
print("NumPy:", np.__version__)

# -------------------------------
# Video file setup
# -------------------------------
video_path = "video.mp4"  # Input video path

import os
if os.path.exists(video_path):
    print("Video found:", video_path)
else:
    raise FileNotFoundError(f"Video not found at {video_path}")

# -------------------------------
# Load YOLOv8 model
# -------------------------------
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 weights

# -------------------------------
# Count Football Touches with YOLO + ByteTrack
# -------------------------------
results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    show=False,
    stream=True
)

ball_touch_count = 0
tracked_ball_id = None
last_position = None
TOUCH_THRESHOLD = 10
MIN_BOX_SIZE = 10

for r in results:
    if r.boxes is None or r.boxes.id is None:
        continue
    for box, cls, obj_id in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):
        if int(cls) == 32:  # Sports ball
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
                continue

            current_position = ((x1 + x2)/2, (y1 + y2)/2)

            if tracked_ball_id != obj_id or last_position is None or \
               math.hypot(current_position[0]-last_position[0], current_position[1]-last_position[1]) > TOUCH_THRESHOLD:
                ball_touch_count += 1
                tracked_ball_id = obj_id
                last_position = current_position

print("Estimated Football Touches:", ball_touch_count)

# -------------------------------
# Foot Contact & Rotation Analysis using MediaPipe
# -------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

left_points = [
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_KNEE
]
right_points = [
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_KNEE
]

DIST_THRESHOLD = 165
MERGE_FRAMES = 2

cap = cv2.VideoCapture(video_path)
frame_idx = 0

left_contact_frames = []
right_contact_frames = []
prev_ball_center = None
prev_hip_center = None
touch_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    results_yolo = model(frame, verbose=False)

    # Detect football
    ball_center = None
    for r in results_yolo:
        if r.boxes is None:
            continue
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) == 32:
                x1, y1, x2, y2 = box
                ball_center = ((x1 + x2)/2, (y1 + y2)/2)
                cv2.circle(frame, (int(ball_center[0]), int(ball_center[1])), 8, (0,255,255), -1)
                break
        if ball_center:
            break
    if ball_center is None:
        continue

    # Check for left/right foot touch
    left_touch = False
    right_touch = False
    if results_pose.pose_landmarks:
        for pt in left_points:
            lm = results_pose.pose_landmarks.landmark[pt]
            px, py = int(lm.x*w), int(lm.y*h)
            if math.hypot(px-ball_center[0], py-ball_center[1]) < DIST_THRESHOLD:
                left_touch = True
        for pt in right_points:
            lm = results_pose.pose_landmarks.landmark[pt]
            px, py = int(lm.x*w), int(lm.y*h)
            if math.hypot(px-ball_center[0], py-ball_center[1]) < DIST_THRESHOLD:
                right_touch = True

    if left_touch:
        left_contact_frames.append(frame_idx)
    if right_touch:
        right_contact_frames.append(frame_idx)

    # Calculate rotation
    rotation = "N/A"
    if prev_ball_center:
        dx = ball_center[0] - prev_ball_center[0]
        rotation = "Forward" if dx > 0 else "Backward"
    prev_ball_center = ball_center

    # Calculate player velocity
    velocity = 0
    if results_pose.pose_landmarks:
        left_hip = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_center = ((left_hip.x+right_hip.x)/2 * w, (left_hip.y+right_hip.y)/2 * h)
        if prev_hip_center:
            dx = hip_center[0] - prev_hip_center[0]
            dy = hip_center[1] - prev_hip_center[1]
            velocity = math.hypot(dx, dy) * 30  # FPS = 30
        prev_hip_center = hip_center

    if left_touch or right_touch:
        touch_data.append({
            "frame": frame_idx,
            "left_touch": int(left_touch),
            "right_touch": int(right_touch),
            "rotation": rotation,
            "velocity": int(velocity)
        })

    if frame_idx % 30 == 0:
        cv2_imshow(frame)

cap.release()

# -------------------------------
# Merge consecutive frames to count touches
# -------------------------------
def merge_frames(frames, merge_len=MERGE_FRAMES):
    if not frames:
        return 0
    frames.sort()
    touches = 1
    last_frame = frames[0]
    for f in frames[1:]:
        if f - last_frame > merge_len:
            touches += 1
        last_frame = f
    return touches

left_leg_touches = merge_frames(left_contact_frames)
right_leg_touches = merge_frames(right_contact_frames)
total_touches = left_leg_touches + right_leg_touches

print("Total Ball Touches:", total_touches)
print("Left Leg Touches:", left_leg_touches)
print("Right Leg Touches:", right_leg_touches)
