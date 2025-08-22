# ⚽ Football Tracker: Ball Hit & Motion Analysis
Real-time football tracking system analyzing ball hits, foot contacts, rotation, and player velocity using YOLOv8 and MediaPipe.

Features

Count total ball hits in a video.
Track left-leg and right-leg touches separately.
Detect ball rotation direction (Forward, Backward, Stable).
Calculate player velocity based on hip movement.
Annotated video output (output.mp4) with all analytics.

---
Technologies Used

Python 3
PyTorch
YOLOv8 (Ultralytics)
MediaPipe (Pose Estimation)
OpenCV
NumPy
---
Installation

```bash
git clone <>
cd FootballTracker
pip install ultralytics opencv-python mediapipe numpy matplotlib

```
Download YOLOv8 pre-trained weights (yolov8n.pt) into the project folder

---
Usage

1. Add your input video (video.mp4) in the project directory.
2. Run the main script:
```bash
python FootBallTrack.py
```

3.Output: output.mp4 annotated with:
Total ball touches
Left/Right leg touches
Ball rotation (Forward/Backward/Stable)
Player velocity (px/s)

---

How It Works

Ball Detection: YOLOv8 detects the football.
Pose Estimation: MediaPipe identifies player foot landmarks.
Contact Analysis: Computes distance between feet and ball for touches.
Rotation & Velocity: Tracks ball movement direction and player speed.
Visualization: Overlays analytics on the output video.

---

## File Structure

| File Name | Description |
|-----------|-------------|
| `FootBallTrack.py` | Main project script |
| `video.mp4`        | Input video |
| `output.mp4`       | Annotated output video |
| `yolov8n.pt`       | YOLOv8 pre-trained weights |
| `README.md`        | Project description |

