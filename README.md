# Drone Detection System - Milestone 1

## Overview

This is Milestone 1 of the Drone Detection System, focusing on **Camera Module** and **Detection** capabilities. The system uses PyQt5 for the user interface and YOLOv8 for object detection.

## Features

- **Camera Module**: Supports RTSP streams, USB cameras, and test mode with video files
- **YOLOv8 Detection**: Detects drones, balloons, and human shapes
- **Color Classification**: Classifies balloon colors for whitelist/blacklist determination
- **Dual UI Screens**: 
  - Operator View: Live video feed with detection visualization
  - System View: Status, alerts, and configuration display

## Requirements

- Python 3.8+
- PyQt5
- OpenCV
- Ultralytics YOLOv8
- See `requirements.txt` for full list

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root (see `.env.example` below)

4. Download YOLOv8 model (will be downloaded automatically on first run):
   - The model will be downloaded from Ultralytics if not present
   - Default: `yolov8n.pt` (nano model)

## Configuration

Create a `.env` file with the following settings:

```env
# Camera Configuration
CAMERA_TYPE=rtsp
CAMERA_RTSP_URL=rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
CAMERA_USB_INDEX=0
CAMERA_FPS=30
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080

# Test Mode Configuration
TEST_OPTION=true
TEST_VIDEO_PATH=test_videos/sample.mp4

# Detection Configuration
YOLO_MODEL_PATH=models/yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.25
YOLO_IOU_THRESHOLD=0.45

# Balloon Color Classification
BALLOON_WHITELIST_COLORS=white,red,green,blue
BALLOON_BLACKLIST_COLORS=black,orange,yellow

# Detection Classes
DETECT_PERSON=true
DETECT_DRONE=true
DETECT_BALLOON=true

# UI Configuration
UI_REFRESH_RATE=30
UI_SHOW_FPS=true
```

### Test Mode

Set `TEST_OPTION=true` to use a video file instead of a live camera stream. Place your test video at the path specified in `TEST_VIDEO_PATH`.

## Usage

Run the application:

```bash
python main.py
```

### UI Navigation

- **Operator View Tab**: Shows live video feed with:
  - Detection bounding boxes (red for blacklist, green for whitelist)
  - Detection labels with confidence scores
  - FPS counter
  - Classification status

- **System View Tab**: Shows:
  - System status (heartbeat, camera, detection)
  - Configuration information
  - Alert log with timestamps

## Project Structure

```
drone-detection-desktop-app/
├── main.py                 # Main application entry point
├── config.py               # Configuration management
├── camera_module.py        # Camera/video handling
├── detection.py            # YOLOv8 detection and color classification
├── ui/
│   ├── main_window.py     # Main window with tabs
│   ├── operator_view.py   # Operator view UI
│   └── system_view.py     # System view UI
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration (create this)
└── README.md              # This file
```

## Detection Classes

The system detects:
- **Person** (COCO class 0): Human shapes
- **Drone**: Custom class (requires custom YOLOv8 model)
- **Balloon**: Custom class with color classification

### Color Classification

Balloons are classified by color:
- **Whitelist**: white, red, green, blue
- **Blacklist**: black, orange, yellow

## Notes

- YOLOv8's default COCO model detects persons but not drones/balloons. For full functionality, you'll need a custom trained model.
- The system is designed to work with custom YOLOv8 models trained on drone/balloon datasets.
- Test mode loops the video file when it reaches the end.

## Future Milestones

- **Milestone 2**: Tracking & Prediction (Kalman Filter/SORT)
- **Milestone 3**: Servo Control via WebSocket
- **Milestone 4**: Alert System (Email/SMS/Push)
- **Milestone 5**: Laser Integration

## Troubleshooting

### Camera Connection Issues

- Check RTSP URL format: `rtsp://username:password@ip:port/path`
- Verify USB camera index (try 0, 1, 2, etc.)
- Ensure camera is accessible on the network (for RTSP)

### Detection Issues

- Ensure YOLOv8 model is downloaded
- Check confidence threshold settings
- Verify test video path if using test mode

### Performance Issues

- Reduce `UI_REFRESH_RATE` for slower systems
- Use YOLOv8n (nano) instead of larger models
- Reduce camera resolution

## License

[Your License Here]

