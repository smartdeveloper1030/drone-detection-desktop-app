# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- Webcam, RTSP camera, or test video file

## Installation Steps

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file:**
   - Copy the example: `cp .env.example .env` (or create manually)
   - Edit `.env` and configure your camera settings
   - Configure `CAMERA_TYPE` (rtsp or usb)
   - For RTSP: Set `CAMERA_RTSP_URL`
   - For USB: Set `CAMERA_USB_INDEX` (usually 0)

## Running the Application

```bash
python main.py
```

## First Run

On first run, YOLOv8 will automatically download the model file (~6MB for nano model). This may take a few minutes depending on your internet connection.

## UI Overview

### Operator View Tab
- **Live Video Feed**: Shows camera/video stream
- **Detection Boxes**: 
  - Red boxes = Blacklist objects (threats)
  - Green boxes = Whitelist objects
- **FPS Counter**: Shows processing frame rate
- **Classification Status**: Shows current detection classification

### System View Tab
- **System Status**: Heartbeat, camera connection, detection status
- **Configuration**: Current settings
- **Alert Log**: System alerts and threat notifications

## Troubleshooting

### "Failed to connect to camera"
- Check camera connection (USB) or network (RTSP)
- Verify RTSP URL format: `rtsp://username:password@ip:port/path`
- Try different USB camera index (0, 1, 2, etc.)

### "Failed to load YOLO model"
- Check internet connection (first-time download)
- Verify `YOLO_MODEL_PATH` in `.env`
- Model will auto-download if using standard YOLOv8 models (yolov8n.pt, etc.)

### Low FPS
- Reduce `UI_REFRESH_RATE` in `.env`
- Use smaller YOLO model (yolov8n instead of yolov8s/m/l/x)
- Reduce camera resolution

## Next Steps

- For custom drone/balloon detection, train a custom YOLOv8 model
- See README.md for detailed configuration options
- Milestone 2 will add tracking and prediction features

