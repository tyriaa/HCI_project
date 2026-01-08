"""
Eye Gaze Detection Web App
===========================
Real-time webcam monitoring to detect if user is looking at the screen.
Flags when user looks away from the screen.
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
import numpy as np
import time
import os
from datetime import datetime
from collections import deque
from ultralytics import YOLO

app = Flask(__name__)

# Global state
gaze_state = {
    'status': 'initializing',
    'direction': 'unknown',
    'is_looking_away': False,
    'flags': [],
    'total_flags': 0,
    'looking_away_duration': 0,
    'last_update': time.time(),
    'pitch': 0,
    'yaw': 0,
    'calibrated': False,
    'pitch_offset': 0,
    'phone_detected': False,
    'phone_flags': 0
}

# Configuration
LOOKING_AWAY_THRESHOLD = 2.0  # seconds before flagging
FLAG_COOLDOWN = 3.0  # seconds between flags

# Calibration - stores baseline pitch when looking at screen
calibration_samples = []
CALIBRATION_FRAMES = 30  # frames to average for calibration

# Load dlib models
detector = None
predictor = None
yolo_model = None

# Phone detection config
PHONE_CLASS_ID = 67  # 'cell phone' in COCO dataset
PHONE_CONFIDENCE_THRESHOLD = 0.5
last_phone_flag_time = 0

def init_models():
    """Initialize dlib face detector, landmark predictor, and YOLO model."""
    global detector, predictor, yolo_model
    
    detector = dlib.get_frontal_face_detector()
    
    # Check for the shape predictor file
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    
    if not os.path.exists(predictor_path):
        print(f"ERROR: {predictor_path} not found!")
        print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Then extract it with: bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
        return False
    
    predictor = dlib.shape_predictor(predictor_path)
    
    # Load YOLO model for phone detection
    print("Loading YOLO model for phone detection...")
    yolo_model = YOLO('yolov8n.pt')  # nano model for speed
    print("✅ YOLO model loaded")
    
    return True


def detect_phone(frame):
    """Detect cell phone in frame using YOLO."""
    if yolo_model is None:
        return False, None
    
    # Run YOLO detection
    results = yolo_model(frame, verbose=False, conf=PHONE_CONFIDENCE_THRESHOLD)
    
    phone_detected = False
    phone_box = None
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == PHONE_CLASS_ID:
                phone_detected = True
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                phone_box = (x1, y1, x2, y2, conf)
                break
    
    return phone_detected, phone_box


def get_eye_region(landmarks, eye_points):
    """Extract eye region coordinates."""
    pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in eye_points])
    return pts


def get_gaze_ratio(eye_points, gray, landmarks):
    """Calculate gaze ratio to determine eye direction."""
    eye_region = get_eye_region(landmarks, eye_points)
    
    # Get bounding box
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    
    # Add padding
    padding = 5
    min_x = max(0, min_x - padding)
    max_x = min(gray.shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(gray.shape[0], max_y + padding)
    
    eye = gray[min_y:max_y, min_x:max_x]
    
    if eye.size == 0:
        return 1.0, "unknown"
    
    # Apply threshold to find pupil
    _, threshold = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY)
    
    height, width = threshold.shape
    left_side = threshold[:, :width // 2]
    right_side = threshold[:, width // 2:]
    
    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)
    
    # Calculate ratio
    if right_white == 0:
        gaze_ratio = 1.0
    else:
        gaze_ratio = left_white / right_white
    
    # Determine direction
    if gaze_ratio < 0.7:
        direction = "right"
    elif gaze_ratio > 1.3:
        direction = "left"
    else:
        direction = "center"
    
    return gaze_ratio, direction


def get_head_pose(landmarks, img_size):
    """Estimate head pose to detect if looking away."""
    # 2D image points
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
        (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
    ], dtype="double")
    
    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # Camera internals
    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get Euler angles
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    
    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    
    return pitch, yaw, roll


def is_looking_at_screen(left_dir, right_dir, pitch, yaw, pitch_offset=0):
    """Determine if user is looking at the screen."""
    # Apply calibration offset to pitch
    adjusted_pitch = pitch - pitch_offset
    
    # Much more lenient thresholds - cameras are often below eye level
    # Yaw: left/right head turn (±35 degrees is generous)
    # Pitch: up/down (±35 degrees, adjusted for camera position)
    YAW_THRESHOLD = 35
    PITCH_UP_THRESHOLD = 35
    PITCH_DOWN_THRESHOLD = 40  # More lenient for looking down (natural position)
    
    # Check head pose first (most reliable)
    if abs(yaw) > YAW_THRESHOLD:
        if yaw > 0:
            return False, "Head turned right"
        else:
            return False, "Head turned left"
    
    if adjusted_pitch > PITCH_DOWN_THRESHOLD:
        return False, "Looking down"
    elif adjusted_pitch < -PITCH_UP_THRESHOLD:
        return False, "Looking up"
    
    # Check eye gaze only if head is centered
    if left_dir == "left" and right_dir == "left":
        return False, "Eyes looking left"
    elif left_dir == "right" and right_dir == "right":
        return False, "Eyes looking right"
    
    return True, "Looking at screen"


# Tracking variables
looking_away_start = None
last_flag_time = 0


def generate_frames():
    """Generate video frames with gaze detection overlay."""
    global gaze_state, looking_away_start, last_flag_time, calibration_samples, last_phone_flag_time
    
    if detector is None or predictor is None:
        if not init_models():
            return
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        current_time = time.time()
        frame_count += 1
        
        # Phone detection (run every 3 frames for performance)
        if frame_count % 3 == 0:
            phone_detected, phone_box = detect_phone(frame)
            gaze_state['phone_detected'] = phone_detected
            
            if phone_detected and phone_box:
                x1, y1, x2, y2, conf = phone_box
                # Draw phone bounding box in orange
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(frame, f"PHONE {conf:.0%}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Flag for phone detection
                if current_time - last_phone_flag_time >= FLAG_COOLDOWN:
                    flag_time = datetime.now().strftime("%H:%M:%S")
                    gaze_state['flags'].append({
                        'time': flag_time,
                        'reason': '📱 Phone detected!',
                        'duration': '-'
                    })
                    gaze_state['total_flags'] += 1
                    gaze_state['phone_flags'] += 1
                    last_phone_flag_time = current_time
        
        # Detect faces
        faces = detector(gray)
        
        if len(faces) == 0:
            # No face detected
            gaze_state['status'] = 'no_face'
            gaze_state['direction'] = 'unknown'
            
            # Draw warning
            cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Consider as looking away
            if looking_away_start is None:
                looking_away_start = current_time
        else:
            face = faces[0]
            landmarks = predictor(gray, face)
            
            # Get eye gaze
            left_eye_points = range(36, 42)
            right_eye_points = range(42, 48)
            
            _, left_dir = get_gaze_ratio(left_eye_points, gray, landmarks)
            _, right_dir = get_gaze_ratio(right_eye_points, gray, landmarks)
            
            # Get head pose
            pitch, yaw, roll = get_head_pose(landmarks, frame.shape)
            
            # Store raw values for display
            gaze_state['pitch'] = pitch
            gaze_state['yaw'] = yaw
            
            # Auto-calibration: collect samples during first frames
            if not gaze_state['calibrated']:
                calibration_samples.append(pitch)
                if len(calibration_samples) >= CALIBRATION_FRAMES:
                    # Use median to avoid outliers
                    gaze_state['pitch_offset'] = np.median(calibration_samples)
                    gaze_state['calibrated'] = True
                    print(f"Calibrated! Pitch offset: {gaze_state['pitch_offset']:.1f}")
                
                # Show calibration message
                cv2.putText(frame, f"CALIBRATING... {len(calibration_samples)}/{CALIBRATION_FRAMES}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                cv2.putText(frame, "Look at the screen normally", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            else:
                # Determine if looking at screen (with calibration offset)
                is_looking, reason = is_looking_at_screen(
                    left_dir, right_dir, pitch, yaw, gaze_state['pitch_offset']
                )
                
                # Update state
                gaze_state['status'] = 'tracking'
                gaze_state['direction'] = reason
                gaze_state['is_looking_away'] = not is_looking
                
                # Draw face rectangle
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                color = (0, 255, 0) if is_looking else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw eye landmarks
                for n in range(36, 48):
                    x_pt = landmarks.part(n).x
                    y_pt = landmarks.part(n).y
                    cv2.circle(frame, (x_pt, y_pt), 2, (255, 255, 0), -1)
                
                # Draw status
                status_text = reason
                status_color = (0, 255, 0) if is_looking else (0, 0, 255)
                cv2.putText(frame, status_text, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Draw head pose info (show adjusted pitch)
                adjusted_pitch = pitch - gaze_state['pitch_offset']
                cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {adjusted_pitch:.1f} (raw: {pitch:.1f})", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Handle looking away detection
                if not is_looking:
                    if looking_away_start is None:
                        looking_away_start = current_time
                    else:
                        duration = current_time - looking_away_start
                        gaze_state['looking_away_duration'] = duration
                        
                        # Check if should flag
                        if duration >= LOOKING_AWAY_THRESHOLD:
                            if current_time - last_flag_time >= FLAG_COOLDOWN:
                                # Add flag
                                flag_time = datetime.now().strftime("%H:%M:%S")
                                gaze_state['flags'].append({
                                    'time': flag_time,
                                    'reason': reason,
                                    'duration': f"{duration:.1f}s"
                                })
                                gaze_state['total_flags'] += 1
                                last_flag_time = current_time
                            
                            # Draw warning
                            cv2.putText(frame, "FLAGGED - LOOK AT SCREEN!", (50, 130),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    looking_away_start = None
                    gaze_state['looking_away_duration'] = 0
        
        # Draw flag counter
        cv2.putText(frame, f"Flags: {gaze_state['total_flags']}", 
                   (frame.shape[1] - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        gaze_state['last_update'] = current_time
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()


@app.route('/')
def index():
    """Render main page."""
    return render_template('eye_gaze_index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """Return current gaze status."""
    return jsonify(gaze_state)


@app.route('/reset_flags')
def reset_flags():
    """Reset flag counter."""
    gaze_state['flags'] = []
    gaze_state['total_flags'] = 0
    return jsonify({'success': True})


@app.route('/recalibrate')
def recalibrate():
    """Reset calibration to recalibrate pitch offset."""
    global calibration_samples
    calibration_samples = []
    gaze_state['calibrated'] = False
    gaze_state['pitch_offset'] = 0
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 50)
    print("Eye Gaze Detection Web App")
    print("=" * 50)
    
    if init_models():
        print("✅ Models loaded successfully")
        print("🌐 Starting server at http://localhost:5001")
        print("=" * 50)
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    else:
        print("❌ Failed to load models. Please download the shape predictor file.")
