# Focus Monitor Pro 🎯

> **Real-time attention monitoring system with AI-powered gaze detection, emotion analysis, and behavioral tracking.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Overview

Focus Monitor Pro is an advanced Human-Computer Interaction (HCI) system that uses computer vision and machine learning to monitor user attention and engagement in real-time. The system employs hierarchical face-iris detection, emotion recognition, and behavioral analysis to provide comprehensive focus tracking.

### Key Features

- **🎯 Hierarchical Gaze Detection**: Face-first, iris-conditional attention tracking with confidence levels
- **😊 Emotion Recognition**: Real-time emotion detection using DeepFace
- **👋 Hand Tracking**: MediaPipe-based hand landmark detection with gesture recognition
- **📱 Object Detection**: YOLOv8-powered detection of phones and distracting objects
- **🚩 Smart Flagging System**: Multi-level flagging for attention loss, tab switches, and suspicious behaviors
- **📊 Live Dashboard**: Modern glassmorphism UI with real-time metrics and visualizations
- **🎵 Emotion-Based Audio**: Dynamic music playback based on detected emotions
- **⚙️ Auto-Calibration**: Automatic pitch offset calibration for personalized tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/macOS/Linux

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tyriaa/HCI_project.git
   cd HCI_project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask opencv-python numpy mediapipe ultralytics deepface pygame tf-keras
   ```

4. **Download required models**
   
   **Windows (PowerShell):**
   ```powershell
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"
   
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
   ```
   
   **macOS/Linux:**
   ```bash
   curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   
   curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

5. **Run the application**
   ```bash
   python eye_gaze_app.py
   ```

6. **Open your browser**
   ```
   http://localhost:5001
   ```

## 🎮 Usage

### Dashboard Controls

- **Reset**: Clear all flags and counters
- **Calibrate**: Recalibrate head pose offset
- **Close**: Shutdown the application with confirmation

### Monitoring Features

The system tracks:
- **Attention Score** (0-100%): Overall engagement level
- **Face Direction**: Head pose orientation
- **Iris Direction**: Eye gaze direction
- **Confidence Level**: Detection reliability (high/low/very_low)
- **Flags**: Attention loss events with timestamps
- **Hands**: Detected hand positions and gestures
- **Objects**: Phones and other distracting items
- **Emotions**: Real-time facial emotion analysis

### Detection Thresholds

| Scenario | Confidence | Threshold | Description |
|----------|-----------|-----------|-------------|
| Face away + Iris away | High | 1.0s | Clear distraction |
| Face away + Iris on screen | Low | 2.0s | Peripheral attention |
| Face forward + Iris away | Very Low | 3.5s | Micro-glances/thinking |
| Face forward + Iris center | High | Instant | Full attention |

## Architecture

```
HCI_project/
├── eye_gaze_app.py          # Main Flask application
├── templates/
│   └── eye_gaze_index.html  # Frontend dashboard
├── face_landmarker.task     # MediaPipe face model
├── hand_landmarker.task     # MediaPipe hand model
├── yolov8n.pt              # YOLOv8 object detection
├── music/                   # Emotion-based audio files
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md
```

## Configuration

### Detection Parameters

Edit `eye_gaze_app.py` to customize:

```python
# Duration thresholds (seconds)
DURATION_T1 = 1.0   # High confidence threshold
DURATION_T2 = 2.0   # Low confidence threshold
DURATION_T3 = 3.5   # Very low confidence threshold

# Head pose thresholds (degrees)
HEAD_YAW_THRESHOLD = 15.0
HEAD_PITCH_THRESHOLD = 12.0

# Iris position thresholds (0-1 scale)
IRIS_H_THRESHOLD = 0.15
IRIS_V_THRESHOLD = 0.15

# Flagging parameters
LOOKING_AWAY_THRESHOLD = 3.0  # Seconds before flagging
FLAG_COOLDOWN = 5.0           # Cooldown between flags
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/video_feed` | GET | Video stream |
| `/status` | GET | Current state JSON |
| `/reset_flags` | GET | Clear all flags |
| `/recalibrate` | GET | Reset calibration |
| `/tab_switch` | POST | Track tab visibility |
| `/window_focus` | POST | Track window focus |
| `/shutdown` | POST | Stop server |

## 🛠️ Technologies

- **Backend**: Flask, Python 3.8+
- **Computer Vision**: OpenCV, MediaPipe
- **Object Detection**: YOLOv8 (Ultralytics)
- **Emotion Recognition**: DeepFace
- **Frontend**: HTML5, TailwindCSS, Vanilla JavaScript
- **Audio**: Pygame

## 📈 Performance

- **FPS**: ~30 FPS on modern hardware
- **Latency**: <50ms detection latency
- **Accuracy**: 95%+ face detection, 90%+ gaze accuracy

## 🐛 Troubleshooting

### Camera not launching
- Check webcam permissions
- Ensure no other application is using the camera
- Verify OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`

### Models not loading
- Re-download model files using the commands above
- Check file sizes: `face_landmarker.task` (~3.6MB), `hand_landmarker.task` (~7.5MB)

### High CPU usage
- Reduce detection frequency in code (increase frame skip)
- Lower webcam resolution
- Disable emotion detection if not needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Clément Frerebeau** - [GitHub](https://github.com/tyriaa)
- **Marc Habib** - [GitHub](https://github.com/Marc-Habib)

## 🙏 Acknowledgments

- MediaPipe team for face and hand tracking models
- Ultralytics for YOLOv8
- DeepFace for emotion recognition
- Flask community for the web framework

---

**⚠️ Note**: This system is designed for educational and research purposes. Ensure compliance with privacy regulations when deploying in production environments.