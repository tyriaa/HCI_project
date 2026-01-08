"""
Eye Gaze Detection Web App
===========================
Real-time webcam monitoring to detect if user is looking at the screen.
Flags when user looks away from the screen.
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
import os
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
import pygame

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
    'phone_flags': 0,
    'tab_switches': 0,
    'window_blurs': 0,
    'tab_switch_flags': 0,
    'is_tab_visible': True,
    'hands_detected': 0,
    'hand_holding_object': False,
    'hand_flags': 0,
    'left_hand_visible': False,
    'right_hand_visible': False,
    'hand_landmarks': [],
    'fingers_extended': {'left': [], 'right': []},
    'hand_gesture': 'none',
    'suspicious_behaviors': [],
    'behavior_flags': 0,
    'eye_openness_left': 100,
    'eye_openness_right': 100,
    'eye_openness_avg': 100,
    'is_blinking': False,
    'blink_count': 0,
    'iris_position': {'left': {'h': 0, 'v': 0}, 'right': {'h': 0, 'v': 0}},
    'head_roll': 0,
    # Full-face attention monitoring
    'attention_score': 100,
    'attention_status': 'attentive',
    'facing_score': 100,
    'occlusion_detected': False,
    'mouth_activity': 0,
    # Emotion detection
    'emotion': 'unknown',
    'emotion_confidence': 0,
    'all_emotions': {}
}

def reset_all_counters():
    """Reset all counters to 0 at app start."""
    global gaze_state, calibration_samples, pose_history, last_good_pose
    global landmark_history, mouth_history, distraction_start_time, attention_state
    
    gaze_state['flags'] = []
    gaze_state['total_flags'] = 0
    gaze_state['looking_away_duration'] = 0
    gaze_state['phone_flags'] = 0
    gaze_state['tab_switches'] = 0
    gaze_state['window_blurs'] = 0
    gaze_state['tab_switch_flags'] = 0
    gaze_state['hand_flags'] = 0
    gaze_state['behavior_flags'] = 0
    gaze_state['blink_count'] = 0
    gaze_state['calibrated'] = False
    gaze_state['pitch_offset'] = 0
    
    # Reset attention monitoring
    gaze_state['attention_score'] = 100
    gaze_state['attention_status'] = 'attentive'
    gaze_state['facing_score'] = 100
    gaze_state['occlusion_detected'] = False
    gaze_state['mouth_activity'] = 0
    
    calibration_samples = []
    # Reset pose smoothing history
    pose_history = {'yaw': [], 'pitch': [], 'roll': []}
    last_good_pose = {'yaw': 0, 'pitch': 0, 'roll': 0}
    
    # Reset attention tracking history
    landmark_history = []
    mouth_history = []
    distraction_start_time = None
    
    print("✅ All counters reset to 0")

# Configuration
LOOKING_AWAY_THRESHOLD = 2.0  # seconds before flagging
FLAG_COOLDOWN = 3.0  # seconds between flags

# Calibration - stores baseline pitch when looking at screen
calibration_samples = []
CALIBRATION_FRAMES = 30  # frames to average for calibration

# Load models
face_landmarker = None
hand_landmarker = None
yolo_model = None

# Phone detection config
PHONE_CLASS_ID = 67  # 'cell phone' in COCO dataset
PHONE_CONFIDENCE_THRESHOLD = 0.5
last_phone_flag_time = 0

def init_models():
    """Initialize MediaPipe face landmarker, hand landmarker, and YOLO."""
    global face_landmarker, hand_landmarker, yolo_model
    
    # Suppress MediaPipe warnings and stderr during initialization
    import warnings
    import sys
    warnings.filterwarnings('ignore')
    
    # Temporarily redirect stderr to suppress MediaPipe cleanup errors
    class SuppressStderr:
        def __enter__(self):
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.close()
            sys.stderr = self._original_stderr
    
    with SuppressStderr():
        # Initialize MediaPipe Face Landmarker
        print("Loading MediaPipe Face Landmarker...")
        try:
            base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(options)
            time.sleep(0.2)
            print("✅ MediaPipe Face Landmarker loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Face Landmarker: {e}")
            return False
        
        # Initialize MediaPipe Hand Landmarker
        print("Loading MediaPipe Hand Landmarker...")
        try:
            hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            time.sleep(0.2)
            print("✅ MediaPipe Hand Landmarker loaded")
        except Exception as e:
            print(f"⚠️ Warning: Hand Landmarker failed to load: {e}")
            print("⚠️ Continuing without hand detection...")
            hand_landmarker = None
    
    # Load YOLO model for phone detection (outside stderr suppression)
    print("Loading YOLO model for phone detection...")
    try:
        yolo_model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All models initialized successfully!")
    print("=" * 50)
    return True


def detect_objects(frame):
    """
    Detect potentially suspicious held objects using YOLO.
    Returns phone detection status and any other held objects.
    
    YOLO COCO classes for suspicious objects:
    - 67: cell phone
    - 63: laptop
    - 64: mouse
    - 65: remote
    - 66: keyboard
    - 73: book
    - 74: clock
    - 39: bottle
    - 41: cup
    - 76: scissors
    """
    if yolo_model is None:
        return False, None, []
    
    phone_detected = False
    phone_box = None
    other_objects = []
    
    # Object classes that could be suspicious if held
    SUSPICIOUS_OBJECT_IDS = {
        67: 'cell phone',
        63: 'laptop', 
        64: 'mouse',
        65: 'remote',
        73: 'book',
        74: 'clock',
        76: 'scissors',
        39: 'bottle',
        41: 'cup',
        84: 'book',
    }
    
    # Run YOLO detection
    results = yolo_model(frame, verbose=False, conf=PHONE_CONFIDENCE_THRESHOLD)
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            
            # Check if detected object is a cell phone
            if class_id == PHONE_CLASS_ID:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                phone_box = (x1, y1, x2, y2, conf)
            
            # Check for other suspicious objects
            elif class_id in SUSPICIOUS_OBJECT_IDS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                obj_name = SUSPICIOUS_OBJECT_IDS[class_id]
                other_objects.append({
                    'name': obj_name,
                    'box': (x1, y1, x2, y2),
                    'confidence': conf
                })
    
    return phone_detected, phone_box, other_objects


def analyze_finger_state(hand_landmarks):
    """Analyze which fingers are extended vs curled."""
    # Landmark indices for each finger
    # Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
    
    fingers_extended = []
    
    # Thumb (special case - check horizontal distance)
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_mcp = hand_landmarks[2]
    thumb_extended = abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x)
    fingers_extended.append(thumb_extended)
    
    # Other fingers (check if tip is above PIP joint)
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_pips = [6, 10, 14, 18]
    
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        # Extended if tip Y is less than PIP Y (tip is higher up)
        extended = tip.y < pip.y
        fingers_extended.append(extended)
    
    return fingers_extended


def detect_hand_gesture(fingers_extended):
    """Detect common hand gestures based on finger states."""
    # fingers_extended = [thumb, index, middle, ring, pinky]
    
    if not any(fingers_extended):
        return "closed_fist"
    elif all(fingers_extended):
        return "open_palm"
    elif fingers_extended == [False, True, False, False, False]:
        return "pointing"
    elif fingers_extended == [True, True, False, False, False]:
        return "peace_sign"
    elif fingers_extended == [False, True, True, False, False]:
        return "two_fingers"
    elif sum(fingers_extended) <= 2:
        return "gripping"
    else:
        return "partial_open"


def detect_hands_holding_object(hand_result, phone_box, face_detected=False, frame_shape=(480, 640)):
    """Detect hands with detailed finger tracking and object-holding detection."""
    if not hand_result.hand_landmarks:
        return False, 0, [], {}
    
    num_hands = len(hand_result.hand_landmarks)
    holding_object = False
    hand_details = {
        'left_hand': None,
        'right_hand': None,
        'gestures': []
    }
    all_landmarks = []
    
    # Process each detected hand
    for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
        # Analyze finger states
        fingers_extended = analyze_finger_state(hand_landmarks)
        gesture = detect_hand_gesture(fingers_extended)
        
        # Get hand position
        wrist = hand_landmarks[0]
        hand_center_x = wrist.x
        hand_center_y = wrist.y
        
        # Determine if left or right hand (based on handedness if available)
        handedness = 'unknown'
        if hand_result.handedness and idx < len(hand_result.handedness):
            handedness = hand_result.handedness[idx][0].category_name.lower()
        
        hand_info = {
            'position': (hand_center_x, hand_center_y),
            'fingers_extended': fingers_extended,
            'gesture': gesture,
            'handedness': handedness
        }
        
        if handedness == 'left':
            hand_details['left_hand'] = hand_info
        elif handedness == 'right':
            hand_details['right_hand'] = hand_info
        
        hand_details['gestures'].append(gesture)
        
        # Store landmarks for visualization
        all_landmarks.append(hand_landmarks)
        
        # ONLY detect holding object if YOLO detected a phone nearby
        # A closed fist alone should NOT trigger "holding object"
        if phone_box:
            h, w = frame_shape[:2]
            px1, py1, px2, py2, _ = phone_box
            phone_center_x = ((px1 + px2) / 2) / w
            phone_center_y = ((py1 + py2) / 2) / h
            
            distance = ((hand_center_x - phone_center_x)**2 + (hand_center_y - phone_center_y)**2)**0.5
            
            if distance < 0.15:  # Hand within 15% of phone
                holding_object = True
        
        # Detection Method 3: Hand in suspicious position (center of screen)
        if 0.25 < hand_center_x < 0.75 and 0.2 < hand_center_y < 0.7:
            if gesture in ['closed_fist', 'gripping', 'partial_open']:
                holding_object = True
    
    return holding_object, num_hands, all_landmarks, hand_details


def detect_suspicious_behaviors(hand_details, face_landmarks, frame_shape):
    """Detect suspicious behaviors like chin holding, hair touching, face touching."""
    behaviors = []
    
    if not face_landmarks or not hand_details:
        return behaviors
    
    h, w = frame_shape[:2]
    
    # Get face bounding box
    face_x_coords = [lm.x for lm in face_landmarks]
    face_y_coords = [lm.y for lm in face_landmarks]
    face_min_x, face_max_x = min(face_x_coords), max(face_x_coords)
    face_min_y, face_max_y = min(face_y_coords), max(face_y_coords)
    
    # Key face landmarks
    nose_tip = face_landmarks[1]
    chin = face_landmarks[152]
    left_cheek = face_landmarks[234]
    right_cheek = face_landmarks[454]
    forehead = face_landmarks[10]
    
    # Check each hand
    for hand_side in ['left_hand', 'right_hand']:
        hand_info = hand_details.get(hand_side)
        if not hand_info:
            continue
        
        position = hand_info.get('position')
        if not position:
            continue
        hand_x, hand_y = position
        gesture = hand_info.get('gesture', 'unknown')
        
        # Behavior 1: Chin holding/resting
        # Hand near chin area with partial closure
        chin_distance = ((hand_x - chin.x)**2 + (hand_y - chin.y)**2)**0.5
        if chin_distance < 0.15 and gesture in ['gripping', 'partial_open', 'closed_fist']:
            behaviors.append('chin_holding')
        
        # Behavior 2: Face touching
        # Hand in face region (between forehead and chin)
        if face_min_x - 0.1 < hand_x < face_max_x + 0.1:
            if face_min_y - 0.1 < hand_y < face_max_y + 0.05:
                # Check specific regions
                
                # Cheek touching
                left_cheek_dist = ((hand_x - left_cheek.x)**2 + (hand_y - left_cheek.y)**2)**0.5
                right_cheek_dist = ((hand_x - right_cheek.x)**2 + (hand_y - right_cheek.y)**2)**0.5
                if min(left_cheek_dist, right_cheek_dist) < 0.12:
                    behaviors.append('face_touching')
                
                # Nose/mouth touching
                nose_distance = ((hand_x - nose_tip.x)**2 + (hand_y - nose_tip.y)**2)**0.5
                if nose_distance < 0.1:
                    behaviors.append('face_touching')
        
        # Behavior 3: Hair touching/adjusting
        # Hand above forehead or at sides of head
        if hand_y < forehead.y - 0.05:  # Above forehead
            behaviors.append('hair_touching')
        elif hand_y < face_min_y + 0.1:  # Near top of head
            # Check if hand is at sides (near ears)
            if hand_x < face_min_x or hand_x > face_max_x:
                behaviors.append('hair_touching')
        
        # Behavior 4: Ear touching
        # Hand near ear region (sides of head at mid-height)
        ear_y = (face_min_y + face_max_y) / 2
        if abs(hand_y - ear_y) < 0.1:
            if hand_x < face_min_x - 0.05 or hand_x > face_max_x + 0.05:
                behaviors.append('ear_touching')
        
        # Behavior 5: Neck touching
        # Hand below chin
        if hand_y > chin.y and hand_y < chin.y + 0.15:
            if face_min_x - 0.1 < hand_x < face_max_x + 0.1:
                behaviors.append('neck_touching')
    
    # Remove duplicates
    behaviors = list(set(behaviors))
    return behaviors


# MediaPipe landmark indices for eyes - comprehensive mapping
# Eye contour landmarks for stable eye region detection
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Iris landmarks (center + 4 points around iris)
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]  # 468 = center
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]  # 473 = center

# Eye landmarks for blink/openness detection (upper and lower eyelid)
LEFT_EYE_UPPER = [159, 145]  # Upper eyelid points
LEFT_EYE_LOWER = [23, 27]    # Lower eyelid points  
RIGHT_EYE_UPPER = [386, 374]  # Upper eyelid points
RIGHT_EYE_LOWER = [253, 257]  # Lower eyelid points

# More precise eye landmarks for EAR (Eye Aspect Ratio) calculation
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # P1-P6 for left eye
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # P1-P6 for right eye

# Vertical eye landmarks for precise openness
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Use MediaPipe's built-in FACEMESH_FACE_OVAL for proper face contour
# This is a frozenset of (i, j) connection pairs
try:
    FACEMESH_FACE_OVAL = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
except AttributeError:
    # Fallback if not available - manual face oval connections
    FACEMESH_FACE_OVAL = frozenset([
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
    ])

# Extract unique indices from face oval connections
FACE_OVAL_INDICES = sorted({idx for conn in FACEMESH_FACE_OVAL for idx in conn})

# Blink detection thresholds
EYE_AR_THRESH = 0.2  # Eye aspect ratio threshold for blink
EYE_AR_CONSEC_FRAMES = 2  # Consecutive frames for blink confirmation
BLINK_COUNTER = 0
TOTAL_BLINKS = 0


def calculate_eye_aspect_ratio(landmarks, eye_indices, img_shape):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Where p1-p6 are the 6 eye landmarks.
    """
    h, w = img_shape[:2]
    
    # Get eye landmark coordinates
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    
    # Calculate distances
    # Vertical distances
    v1 = np.sqrt((pts[1][0] - pts[5][0])**2 + (pts[1][1] - pts[5][1])**2)
    v2 = np.sqrt((pts[2][0] - pts[4][0])**2 + (pts[2][1] - pts[4][1])**2)
    
    # Horizontal distance
    h_dist = np.sqrt((pts[0][0] - pts[3][0])**2 + (pts[0][1] - pts[3][1])**2)
    
    if h_dist == 0:
        return 0.3  # Default open eye value
    
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear


def calculate_eye_openness(landmarks, img_shape):
    """
    Calculate eye openness percentage for both eyes.
    Returns tuple of (left_openness, right_openness, avg_openness, is_blinking)
    """
    h, w = img_shape[:2]
    
    # Get vertical eye distances
    left_top = landmarks[LEFT_EYE_TOP]
    left_bottom = landmarks[LEFT_EYE_BOTTOM]
    right_top = landmarks[RIGHT_EYE_TOP]
    right_bottom = landmarks[RIGHT_EYE_BOTTOM]
    
    # Calculate vertical distances (in pixels)
    left_height = abs(left_bottom.y - left_top.y) * h
    right_height = abs(right_bottom.y - right_top.y) * h
    
    # Get horizontal eye width for normalization
    left_width = abs(landmarks[33].x - landmarks[133].x) * w
    right_width = abs(landmarks[362].x - landmarks[263].x) * w
    
    # Calculate openness ratio (height/width)
    left_openness = left_height / max(left_width, 1) if left_width > 0 else 0
    right_openness = right_height / max(right_width, 1) if right_width > 0 else 0
    
    avg_openness = (left_openness + right_openness) / 2
    
    # Normalize to 0-100% (typical open eye ratio is ~0.25-0.35)
    left_pct = min(100, max(0, (left_openness / 0.35) * 100))
    right_pct = min(100, max(0, (right_openness / 0.35) * 100))
    avg_pct = (left_pct + right_pct) / 2
    
    # Detect blink (both eyes closed)
    is_blinking = avg_openness < 0.15
    
    return left_pct, right_pct, avg_pct, is_blinking


def get_iris_position(landmarks, img_shape):
    """
    Get precise iris position relative to eye corners.
    Returns normalized position (-1 to 1) for both horizontal and vertical.
    """
    h, w = img_shape[:2]
    
    # Left eye iris
    left_iris_center = landmarks[468]
    left_eye_inner = landmarks[133]
    left_eye_outer = landmarks[33]
    
    # Right eye iris
    right_iris_center = landmarks[473]
    right_eye_inner = landmarks[362]
    right_eye_outer = landmarks[263]
    
    # Calculate left eye iris position
    left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
    left_eye_center_x = (left_eye_outer.x + left_eye_inner.x) / 2
    left_iris_h = (left_iris_center.x - left_eye_center_x) / (left_eye_width / 2) if left_eye_width > 0 else 0
    
    # Vertical position for left eye
    left_eye_top = landmarks[LEFT_EYE_TOP]
    left_eye_bottom = landmarks[LEFT_EYE_BOTTOM]
    left_eye_height = abs(left_eye_bottom.y - left_eye_top.y)
    left_eye_center_y = (left_eye_top.y + left_eye_bottom.y) / 2
    left_iris_v = (left_iris_center.y - left_eye_center_y) / (left_eye_height / 2) if left_eye_height > 0 else 0
    
    # Calculate right eye iris position
    right_eye_width = abs(right_eye_inner.x - right_eye_outer.x)
    right_eye_center_x = (right_eye_outer.x + right_eye_inner.x) / 2
    right_iris_h = (right_iris_center.x - right_eye_center_x) / (right_eye_width / 2) if right_eye_width > 0 else 0
    
    # Vertical position for right eye
    right_eye_top = landmarks[RIGHT_EYE_TOP]
    right_eye_bottom = landmarks[RIGHT_EYE_BOTTOM]
    right_eye_height = abs(right_eye_bottom.y - right_eye_top.y)
    right_eye_center_y = (right_eye_top.y + right_eye_bottom.y) / 2
    right_iris_v = (right_iris_center.y - right_eye_center_y) / (right_eye_height / 2) if right_eye_height > 0 else 0
    
    return {
        'left': {'horizontal': left_iris_h, 'vertical': left_iris_v},
        'right': {'horizontal': right_iris_h, 'vertical': right_iris_v},
        'avg_horizontal': (left_iris_h + right_iris_h) / 2,
        'avg_vertical': (left_iris_v + right_iris_v) / 2
    }


def get_gaze_direction(landmarks, img_shape, head_yaw=0, head_pitch=0):
    """
    Detect gaze direction using MediaPipe face landmarks.
    Combines head pose with iris position for accurate detection.
    
    All directions are from the HUMAN's perspective (not camera).
    The video feed is mirrored like a mirror.
    
    Args:
        landmarks: MediaPipe face landmarks
        img_shape: frame shape (h, w, c)
        head_yaw: head yaw angle from get_head_pose
        head_pitch: head pitch angle from get_head_pose
    
    Returns:
        dict with direction info
    """
    h, w = img_shape[:2]
    
    # === GET NOSE TIP AS FACE CENTER REFERENCE ===
    nose_tip = landmarks[1]
    
    # === GET FACE BOUNDING LANDMARKS ===
    # Left side of face (from camera view)
    left_face = landmarks[234]   # Left cheek
    # Right side of face (from camera view)  
    right_face = landmarks[454]  # Right cheek
    # Top of face
    forehead = landmarks[10]
    # Bottom of face
    chin = landmarks[152]
    
    # Calculate face center
    face_center_x = (left_face.x + right_face.x) / 2
    face_center_y = (forehead.y + chin.y) / 2
    face_width = abs(right_face.x - left_face.x)
    face_height = abs(chin.y - forehead.y)
    
    # === IRIS TRACKING ===
    # Left eye iris (landmarks 468-472, center is 468)
    left_iris = landmarks[468]
    # Right eye iris (landmarks 473-477, center is 473)
    right_iris = landmarks[473]
    
    # Left eye corners
    left_eye_inner = landmarks[133]
    left_eye_outer = landmarks[33]
    # Right eye corners
    right_eye_inner = landmarks[362]
    right_eye_outer = landmarks[263]
    
    # Calculate iris position relative to eye corners
    # For left eye
    left_eye_width = left_eye_inner.x - left_eye_outer.x
    if left_eye_width > 0:
        left_iris_ratio = (left_iris.x - left_eye_outer.x) / left_eye_width
    else:
        left_iris_ratio = 0.5
    
    # For right eye
    right_eye_width = right_eye_outer.x - right_eye_inner.x
    if right_eye_width > 0:
        right_iris_ratio = (right_iris.x - right_eye_inner.x) / right_eye_width
    else:
        right_iris_ratio = 0.5
    
    # Average iris position (0 = looking left, 0.5 = center, 1 = looking right)
    avg_iris_h = (left_iris_ratio + right_iris_ratio) / 2
    
    # Vertical iris tracking
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    
    left_eye_height = left_eye_bottom.y - left_eye_top.y
    right_eye_height = right_eye_bottom.y - right_eye_top.y
    
    if left_eye_height > 0:
        left_iris_v = (left_iris.y - left_eye_top.y) / left_eye_height
    else:
        left_iris_v = 0.5
        
    if right_eye_height > 0:
        right_iris_v = (right_iris.y - right_eye_top.y) / right_eye_height
    else:
        right_iris_v = 0.5
    
    avg_iris_v = (left_iris_v + right_iris_v) / 2
    
    # === DETERMINE DIRECTION ===
    # Use head pose classification (primary, more reliable)
    head_direction = classify_head_direction(head_yaw, head_pitch)
    
    direction = "center"
    source = ""
    
    # Priority 1: Head pose (more reliable for large movements)
    if head_direction != "center":
        direction = head_direction
        source = "head"
    else:
        # Priority 2: Iris position (for subtle eye movements when head is centered)
        IRIS_H_THRESHOLD = 0.12      # deviation from center (0.5)
        IRIS_V_THRESHOLD = 0.12      # deviation from center (0.5)
        
        iris_h_offset = avg_iris_h - 0.5  # Positive = iris moved right in image
        iris_v_offset = avg_iris_v - 0.5  # Positive = iris moved down
        
        # In mirrored video: iris moving right in image = user looking to THEIR left
        if iris_h_offset > IRIS_H_THRESHOLD:
            direction = "left"
            source = "eyes"
        elif iris_h_offset < -IRIS_H_THRESHOLD:
            direction = "right"
            source = "eyes"
        elif iris_v_offset > IRIS_V_THRESHOLD:
            direction = "down"
            source = "eyes"
        elif iris_v_offset < -IRIS_V_THRESHOLD:
            direction = "up"
            source = "eyes"
    
    return {
        'direction': direction,
        'source': source,
        'iris_h': avg_iris_h,
        'iris_v': avg_iris_v,
        'head_yaw': head_yaw,
        'head_pitch': head_pitch,
        'is_center': direction == "center"
    }


# ============================================================
# HEAD POSE ESTIMATION WITH SMOOTHING
# ============================================================

# Smoothing buffers for head pose (EMA / moving average)
pose_history = {
    'yaw': [],
    'pitch': [],
    'roll': []
}
POSE_SMOOTHING_FRAMES = 10  # Number of frames for moving average
EMA_ALPHA = 0.3  # Exponential moving average alpha (0.1-0.5, lower = smoother)

# Last known good pose (for bad frame rejection)
last_good_pose = {'yaw': 0, 'pitch': 0, 'roll': 0}

# 6-point landmark indices for head pose (standard set)
POSE_LANDMARK_INDICES = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye outer corner
    263,  # Right eye outer corner
    61,   # Left mouth corner
    291,  # Right mouth corner
]

# Canonical 3D face model coordinates (in mm)
# These are approximate but PnP is tolerant
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -63.6, -12.5),       # Chin
    (-43.3, 32.7, -26.0),      # Left eye outer corner
    (43.3, 32.7, -26.0),       # Right eye outer corner
    (-28.9, -28.9, -24.1),     # Left mouth corner
    (28.9, -28.9, -24.1),      # Right mouth corner
], dtype=np.float64)


def smooth_angle(angle, history_key, use_ema=True):
    """
    Apply smoothing to angle using EMA or moving average.
    
    Args:
        angle: Current angle value
        history_key: Key in pose_history dict ('yaw', 'pitch', 'roll')
        use_ema: If True, use exponential moving average; else use simple moving average
    
    Returns:
        Smoothed angle value
    """
    global pose_history
    
    # Add to history
    pose_history[history_key].append(angle)
    
    # Keep only last N frames
    if len(pose_history[history_key]) > POSE_SMOOTHING_FRAMES:
        pose_history[history_key] = pose_history[history_key][-POSE_SMOOTHING_FRAMES:]
    
    if use_ema:
        # Exponential Moving Average
        if len(pose_history[history_key]) == 1:
            return angle
        
        smoothed = pose_history[history_key][0]
        for val in pose_history[history_key][1:]:
            smoothed = EMA_ALPHA * val + (1 - EMA_ALPHA) * smoothed
        return smoothed
    else:
        # Simple Moving Average
        return np.mean(pose_history[history_key])


def get_head_pose(landmarks, img_shape):
    """
    Estimate head pose using 6-point PnP method with smoothing.
    
    A) Get 2D landmarks from MediaPipe Face Mesh (6 points)
    B) Map to canonical 3D face model
    C) Solve PnP → rotation → yaw/pitch/roll
    D) Smooth angles with EMA
    
    Returns:
        pitch, yaw, roll in degrees (smoothed)
    """
    global last_good_pose
    
    h, w = img_shape[:2]
    
    # A) Extract 2D landmark points
    image_points = np.array([
        (landmarks[idx].x * w, landmarks[idx].y * h) 
        for idx in POSE_LANDMARK_INDICES
    ], dtype=np.float64)
    
    # Check for bad frame (landmarks out of bounds or invalid)
    for pt in image_points:
        if pt[0] < 0 or pt[0] > w or pt[1] < 0 or pt[1] > h:
            # Bad frame - return last good pose
            return last_good_pose['pitch'], last_good_pose['yaw'], last_good_pose['roll']
    
    # B) 3D model points (already defined as FACE_3D_MODEL)
    
    # C) Camera intrinsic parameters
    # Focal length ≈ image width (standard approximation)
    focal_length = w
    center = (w / 2, h / 2)
    
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_3D_MODEL, 
        image_points, 
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return last_good_pose['pitch'], last_good_pose['yaw'], last_good_pose['roll']
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get Euler angles using RQDecomp3x3
    # This gives more stable results than decomposeProjectionMatrix
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    
    pitch = angles[0]  # X-axis rotation (nodding)
    yaw = angles[1]    # Y-axis rotation (shaking head left/right)
    roll = angles[2]   # Z-axis rotation (tilting head)
    
    # D) Apply smoothing
    pitch = smooth_angle(pitch, 'pitch')
    yaw = smooth_angle(yaw, 'yaw')
    roll = smooth_angle(roll, 'roll')
    
    # Update last good pose
    last_good_pose['pitch'] = pitch
    last_good_pose['yaw'] = yaw
    last_good_pose['roll'] = roll
    
    return pitch, yaw, roll


def classify_head_direction(yaw, pitch):
    """
    Classify head direction based on yaw and pitch angles.
    
    Thresholds (from user's perspective, mirrored video):
    - yaw > +15° → looking left (user's left)
    - yaw < -15° → looking right (user's right)
    - pitch > +10° → looking down
    - pitch < -10° → looking up
    - else → forward/center
    
    Returns:
        direction: str ('left', 'right', 'up', 'down', 'center')
    """
    YAW_THRESHOLD = 15.0
    PITCH_THRESHOLD = 10.0
    
    # Check yaw first (horizontal movement is more common)
    if yaw > YAW_THRESHOLD:
        return "left"
    elif yaw < -YAW_THRESHOLD:
        return "right"
    # Then check pitch
    elif pitch > PITCH_THRESHOLD:
        return "down"
    elif pitch < -PITCH_THRESHOLD:
        return "up"
    else:
        return "center"


# ============================================================
# FULL-FACE ATTENTION MONITORING SYSTEM
# ============================================================

# Attention state tracking
attention_state = {
    'facing_score': 100,           # 0-100, how directly facing camera
    'symmetry_score': 100,         # 0-100, face symmetry
    'occlusion_detected': False,   # eyes/face blocked
    'occlusion_regions': [],       # which regions are occluded
    'mouth_open_ratio': 0,         # 0-1, mouth openness
    'mouth_activity': 0,           # 0-100, lip movement intensity
    'landmark_confidence': 100,    # 0-100, landmark stability
    'attention_score': 100,        # 0-100, overall attention
    'attention_status': 'attentive',  # attentive, distracted, away, occluded
    'distraction_duration': 0,     # seconds of continuous distraction
}

# Duration tracking for attention decisions
distraction_start_time = None
DISTRACTION_THRESHOLD_SECONDS = 2.0  # Flag after 2 seconds of distraction

# Landmark jitter tracking for occlusion detection
landmark_history = []
JITTER_HISTORY_FRAMES = 5

# Mouth activity tracking
mouth_history = []
MOUTH_HISTORY_FRAMES = 10


def calculate_facing_score(landmarks, img_shape):
    """
    Calculate how directly the face is facing the camera (0-100).
    Uses multiple signals:
    1. Face symmetry (left vs right landmark distances)
    2. Nose-to-face-center offset
    3. Eye visibility balance
    
    Returns:
        facing_score: 0-100 (100 = directly facing)
    """
    h, w = img_shape[:2]
    
    # Get key landmarks
    nose_tip = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    chin = landmarks[152]
    forehead = landmarks[10]
    
    # 1. Calculate face center
    face_center_x = (left_cheek.x + right_cheek.x) / 2
    face_center_y = (forehead.y + chin.y) / 2
    
    # 2. Nose offset from face center (should be ~0 when facing camera)
    nose_offset_x = abs(nose_tip.x - face_center_x)
    nose_offset_score = max(0, 100 - nose_offset_x * 500)  # Penalize offset
    
    # 3. Face symmetry - compare left and right distances
    left_dist = abs(nose_tip.x - left_cheek.x)
    right_dist = abs(nose_tip.x - right_cheek.x)
    
    if max(left_dist, right_dist) > 0:
        symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
    else:
        symmetry_ratio = 1.0
    symmetry_score = symmetry_ratio * 100
    
    # 4. Eye balance - both eyes should be similarly visible
    left_eye_width = abs(landmarks[133].x - landmarks[33].x)
    right_eye_width = abs(landmarks[263].x - landmarks[362].x)
    
    if max(left_eye_width, right_eye_width) > 0:
        eye_ratio = min(left_eye_width, right_eye_width) / max(left_eye_width, right_eye_width)
    else:
        eye_ratio = 1.0
    eye_balance_score = eye_ratio * 100
    
    # Combine scores (weighted average)
    facing_score = (nose_offset_score * 0.4 + symmetry_score * 0.4 + eye_balance_score * 0.2)
    
    return min(100, max(0, facing_score))


def detect_occlusion(landmarks, img_shape, hand_landmarks_list=None):
    """
    Detect if face regions are occluded (hands, hair, objects).
    
    Checks:
    1. Landmark confidence/jitter (sudden changes indicate occlusion)
    2. Hand proximity to face regions
    3. Eye region visibility
    
    Returns:
        is_occluded: bool
        occluded_regions: list of region names
        confidence: 0-100
    """
    global landmark_history
    
    h, w = img_shape[:2]
    occluded_regions = []
    confidence = 100
    
    # Key face regions to monitor
    left_eye_center = landmarks[468]  # Iris center
    right_eye_center = landmarks[473]
    mouth_center = landmarks[13]  # Upper lip center
    nose_tip = landmarks[1]
    
    # 1. Track landmark jitter (high jitter = possible occlusion)
    current_landmarks = {
        'left_eye': (left_eye_center.x, left_eye_center.y),
        'right_eye': (right_eye_center.x, right_eye_center.y),
        'mouth': (mouth_center.x, mouth_center.y),
        'nose': (nose_tip.x, nose_tip.y)
    }
    
    landmark_history.append(current_landmarks)
    if len(landmark_history) > JITTER_HISTORY_FRAMES:
        landmark_history = landmark_history[-JITTER_HISTORY_FRAMES:]
    
    # Calculate jitter for each region
    if len(landmark_history) >= 3:
        for region in ['left_eye', 'right_eye', 'mouth', 'nose']:
            positions = [lm[region] for lm in landmark_history]
            
            # Calculate variance in positions
            x_vals = [p[0] for p in positions]
            y_vals = [p[1] for p in positions]
            
            x_var = np.var(x_vals) if len(x_vals) > 1 else 0
            y_var = np.var(y_vals) if len(y_vals) > 1 else 0
            
            jitter = (x_var + y_var) * 10000  # Scale up for visibility
            
            # High jitter indicates occlusion
            if jitter > 5:  # Threshold for "suspicious" jitter
                if 'eye' in region:
                    occluded_regions.append('eyes')
                elif region == 'mouth':
                    occluded_regions.append('mouth')
                confidence -= 20
    
    # 2. Check hand proximity to face (if hand landmarks available)
    if hand_landmarks_list:
        for hand_lm in hand_landmarks_list:
            if hand_lm:
                # Get hand center (wrist landmark)
                hand_x = hand_lm[0].x if hasattr(hand_lm[0], 'x') else hand_lm[0][0]
                hand_y = hand_lm[0].y if hasattr(hand_lm[0], 'y') else hand_lm[0][1]
                
                # Check if hand is near face regions
                face_center_x = (landmarks[234].x + landmarks[454].x) / 2
                face_center_y = (landmarks[10].y + landmarks[152].y) / 2
                
                dist_to_face = np.sqrt((hand_x - face_center_x)**2 + (hand_y - face_center_y)**2)
                
                if dist_to_face < 0.15:  # Hand very close to face
                    # Check which region
                    if abs(hand_y - left_eye_center.y) < 0.1:
                        occluded_regions.append('eyes_by_hand')
                    if abs(hand_y - mouth_center.y) < 0.1:
                        occluded_regions.append('mouth_by_hand')
                    confidence -= 30
    
    # Remove duplicates
    occluded_regions = list(set(occluded_regions))
    is_occluded = len(occluded_regions) > 0 or confidence < 70
    
    return is_occluded, occluded_regions, max(0, confidence)


def calculate_mouth_activity(landmarks, img_shape):
    """
    Calculate mouth activity (opening ratio and movement).
    
    Returns:
        mouth_open_ratio: 0-1 (0 = closed, 1 = wide open)
        mouth_activity: 0-100 (movement intensity over time)
    """
    global mouth_history
    
    # Mouth landmarks
    upper_lip = landmarks[13]   # Upper lip center
    lower_lip = landmarks[14]   # Lower lip center
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    
    # Calculate mouth opening
    mouth_height = abs(lower_lip.y - upper_lip.y)
    mouth_width = abs(right_mouth.x - left_mouth.x)
    
    if mouth_width > 0:
        mouth_open_ratio = min(1.0, mouth_height / mouth_width * 2)
    else:
        mouth_open_ratio = 0
    
    # Track mouth state over time for activity detection
    mouth_history.append(mouth_open_ratio)
    if len(mouth_history) > MOUTH_HISTORY_FRAMES:
        mouth_history = mouth_history[-MOUTH_HISTORY_FRAMES:]
    
    # Calculate activity (variance in mouth opening = speaking/movement)
    if len(mouth_history) >= 3:
        mouth_activity = np.std(mouth_history) * 500  # Scale for 0-100
        mouth_activity = min(100, mouth_activity)
    else:
        mouth_activity = 0
    
    return mouth_open_ratio, mouth_activity


def calculate_attention_score(landmarks, img_shape, yaw, pitch, roll, hand_landmarks=None):
    """
    Calculate unified attention score combining all signals.
    
    Primary signals:
    - Face present (handled externally)
    - Head pose (yaw/pitch) duration-based
    - Facing camera score
    
    Secondary signals:
    - Occlusion detection
    - Mouth activity
    
    Returns:
        attention_data: dict with all attention metrics
    """
    global attention_state, distraction_start_time
    
    # 1. Facing camera score
    facing_score = calculate_facing_score(landmarks, img_shape)
    
    # 2. Occlusion detection
    is_occluded, occluded_regions, landmark_confidence = detect_occlusion(
        landmarks, img_shape, hand_landmarks
    )
    
    # 3. Mouth activity
    mouth_open_ratio, mouth_activity = calculate_mouth_activity(landmarks, img_shape)
    
    # 4. Head pose contribution to attention
    # Large yaw/pitch = not paying attention
    yaw_penalty = min(50, abs(yaw) * 2)  # Up to 50 points penalty
    pitch_penalty = min(30, abs(pitch) * 1.5)  # Up to 30 points penalty
    
    # 5. Calculate overall attention score
    base_score = 100
    
    # Apply penalties
    base_score -= yaw_penalty
    base_score -= pitch_penalty
    
    # Facing score contribution
    if facing_score < 70:
        base_score -= (70 - facing_score) * 0.5
    
    # Occlusion penalty
    if is_occluded:
        base_score -= 20
    
    attention_score = max(0, min(100, base_score))
    
    # 6. Determine attention status
    if is_occluded and 'eyes' in str(occluded_regions):
        status = 'occluded'
    elif attention_score < 30:
        status = 'away'
    elif attention_score < 60:
        status = 'distracted'
    else:
        status = 'attentive'
    
    # 7. Track distraction duration
    current_time = time.time()
    if status in ['distracted', 'away', 'occluded']:
        if distraction_start_time is None:
            distraction_start_time = current_time
        distraction_duration = current_time - distraction_start_time
    else:
        distraction_start_time = None
        distraction_duration = 0
    
    # Update attention state
    attention_state.update({
        'facing_score': round(facing_score, 1),
        'symmetry_score': round(facing_score, 1),  # Same as facing for now
        'occlusion_detected': is_occluded,
        'occlusion_regions': occluded_regions,
        'mouth_open_ratio': round(mouth_open_ratio, 2),
        'mouth_activity': round(mouth_activity, 1),
        'landmark_confidence': landmark_confidence,
        'attention_score': round(attention_score, 1),
        'attention_status': status,
        'distraction_duration': round(distraction_duration, 1),
    })
    
    return attention_state


def draw_face_mesh(frame, landmarks, draw_iris=True, draw_contours=True):
    """
    Draw face mesh visualization on frame.
    Uses MediaPipe's FACEMESH_FACE_OVAL for proper face contour.
    Includes iris tracking.
    """
    h, w = frame.shape[:2]
    
    if draw_contours:
        # Draw face oval using proper connection pairs
        for (i, j) in FACEMESH_FACE_OVAL:
            pt1 = (int(landmarks[i].x * w), int(landmarks[i].y * h))
            pt2 = (int(landmarks[j].x * w), int(landmarks[j].y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
        
        # Draw face oval points
        for idx in FACE_OVAL_INDICES:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)
    
    if draw_iris:
        # Draw left iris
        left_iris_center = landmarks[468]
        left_iris_x = int(left_iris_center.x * w)
        left_iris_y = int(left_iris_center.y * h)
        cv2.circle(frame, (left_iris_x, left_iris_y), 3, (0, 255, 255), -1)
        
        # Draw left iris outline
        for i in range(1, 5):
            pt = landmarks[468 + i]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (0, 200, 200), -1)
        
        # Draw right iris
        right_iris_center = landmarks[473]
        right_iris_x = int(right_iris_center.x * w)
        right_iris_y = int(right_iris_center.y * h)
        cv2.circle(frame, (right_iris_x, right_iris_y), 3, (0, 255, 255), -1)
        
        # Draw right iris outline
        for i in range(1, 5):
            pt = landmarks[473 + i]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (0, 200, 200), -1)
    
    # Draw eye contours
    for idx in LEFT_EYE_INDICES:
        pt = landmarks[idx]
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (255, 255, 0), -1)
    
    for idx in RIGHT_EYE_INDICES:
        pt = landmarks[idx]
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (255, 255, 0), -1)
    
    return frame


def is_looking_at_screen(gaze_data):
    """
    Determine if user is looking at the screen based on gaze direction.
    
    Args:
        gaze_data: dict from get_gaze_direction()
    
    Returns:
        (is_looking: bool, reason: str)
    """
    direction = gaze_data.get('direction', 'center')
    source = gaze_data.get('source', '')
    
    if direction == "center":
        return True, "Looking at screen"
    
    # Format direction string from human's perspective
    direction_upper = direction.upper()
    source_text = f" ({source})" if source else ""
    
    return False, f"Looking {direction_upper}{source_text}"


# Tracking variables
looking_away_start = None
last_flag_time = 0
last_emotion_time = 0
EMOTION_DETECTION_INTERVAL = 3.0  # Detect emotion every 3 seconds for performance

# Initialize pygame mixer for audio
pygame.mixer.init()
current_playing_emotion = None

# Music mapping for emotions
EMOTION_MUSIC = {
    'happy': 'music/happy.mp3',
    'sad': 'music/sad.mp3',
    'surprise': 'music/surprise.mp3',
    'disgust': 'music/disgust.mp3',
    'neutral': 'music/neutral.mp3'
}


def play_emotion_music(emotion):
    """Play music based on detected emotion."""
    global current_playing_emotion
    
    # Don't restart if same emotion is already playing
    if current_playing_emotion == emotion and pygame.mixer.music.get_busy():
        return
    
    # Check if music file exists for this emotion
    music_file = EMOTION_MUSIC.get(emotion)
    if music_file and os.path.exists(music_file):
        try:
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play(-1)  # Loop indefinitely
            pygame.mixer.music.set_volume(0.3)  # Set volume to 30%
            current_playing_emotion = emotion
            print(f"🎵 Playing music for emotion: {emotion}")
        except Exception as e:
            print(f"⚠️ Could not play music for {emotion}: {e}")
    else:
        # Stop music if no file exists for this emotion
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            current_playing_emotion = None


def detect_emotion(frame):
    """Detect emotion from face using DeepFace."""
    try:
        # Analyze emotion - use enforce_detection=False to avoid errors when face not found
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        
        if result and len(result) > 0:
            emotions = result[0].get('emotion', {})
            dominant_emotion = result[0].get('dominant_emotion', 'unknown')
            confidence = emotions.get(dominant_emotion, 0)
            return dominant_emotion, confidence, emotions
    except Exception as e:
        pass
    
    return 'unknown', 0, {}


def generate_frames():
    """Generate video frames with gaze detection overlay."""
    global gaze_state, looking_away_start, last_flag_time, calibration_samples, last_phone_flag_time, last_emotion_time
    
    if face_landmarker is None:
        if not init_models():
            return
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_count = 0
    phone_detected = False
    phone_box = None
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        current_time = time.time()
        frame_count += 1
        
        # Object detection (run every 3 frames for performance)
        if frame_count % 3 == 0:
            phone_detected, phone_box, other_objects = detect_objects(frame)
            gaze_state['phone_detected'] = phone_detected
            
            # Draw phone if detected
            if phone_detected and phone_box:
                x1, y1, x2, y2, conf = phone_box
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
            
            # Draw and flag other suspicious objects
            for obj in other_objects:
                x1, y1, x2, y2 = obj['box']
                obj_name = obj['name']
                conf = obj['confidence']
                
                # Draw object bounding box in yellow
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{obj_name.upper()} {conf:.0%}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Flag for object detection (use same cooldown)
                if current_time - last_phone_flag_time >= FLAG_COOLDOWN:
                    flag_time = datetime.now().strftime("%H:%M:%S")
                    gaze_state['flags'].append({
                        'time': flag_time,
                        'reason': f'⚠️ {obj_name.title()} detected!',
                        'duration': '-'
                    })
                    gaze_state['total_flags'] += 1
                    last_phone_flag_time = current_time
        
        # Emotion detection (run every EMOTION_DETECTION_INTERVAL seconds for performance)
        if current_time - last_emotion_time >= EMOTION_DETECTION_INTERVAL:
            emotion, confidence, all_emotions = detect_emotion(frame)
            
            # Filter out fear, angry and sad emotions - replace with next best
            if emotion in ['fear', 'angry', 'sad']:
                # Get the next best emotion that's not fear, angry or sad
                filtered_emotions = {k: v for k, v in all_emotions.items() if k not in ['fear', 'angry', 'sad']}
                if filtered_emotions:
                    emotion = max(filtered_emotions, key=filtered_emotions.get)
                    confidence = filtered_emotions[emotion]
                else:
                    emotion = 'neutral'
                    confidence = 0
            
            gaze_state['emotion'] = emotion
            gaze_state['emotion_confidence'] = confidence
            gaze_state['all_emotions'] = all_emotions
            last_emotion_time = current_time
            
            # Play music based on emotion
            play_emotion_music(emotion)
            
            # Draw emotion on frame
            emotion_color = (0, 255, 0)  # Green by default
            if emotion in ['sad', 'disgust']:
                emotion_color = (0, 0, 255)  # Red for negative emotions
            elif emotion == 'surprise':
                emotion_color = (0, 255, 255)  # Yellow for surprise
            
            cv2.putText(frame, f"Emotion: {emotion} ({confidence:.0f}%)", 
                       (frame.shape[1] - 300, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Hand detection (run every 2 frames for better responsiveness)
        if frame_count % 2 == 0 and hand_landmarker is not None:
            mp_image_hands = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            hand_result = hand_landmarker.detect(mp_image_hands)
            
            # Check if face was detected in this frame
            face_in_frame = detection_result.face_landmarks if 'detection_result' in locals() else False
            holding_object, num_hands, hand_landmarks_list, hand_details = detect_hands_holding_object(
                hand_result, 
                phone_box if phone_detected else None,
                face_detected=bool(face_in_frame),
                frame_shape=frame.shape
            )
            
            # Update global state with detailed hand information
            gaze_state['hands_detected'] = num_hands
            gaze_state['hand_holding_object'] = holding_object
            gaze_state['hand_landmarks'] = hand_landmarks_list
            gaze_state['left_hand_visible'] = hand_details.get('left_hand') is not None
            gaze_state['right_hand_visible'] = hand_details.get('right_hand') is not None
            
            # Update finger states
            left_hand = hand_details.get('left_hand')
            right_hand = hand_details.get('right_hand')
            
            if left_hand and 'fingers_extended' in left_hand:
                gaze_state['fingers_extended']['left'] = left_hand['fingers_extended']
            else:
                gaze_state['fingers_extended']['left'] = []
                
            if right_hand and 'fingers_extended' in right_hand:
                gaze_state['fingers_extended']['right'] = right_hand['fingers_extended']
            else:
                gaze_state['fingers_extended']['right'] = []
            
            # Update gesture
            gestures = hand_details.get('gestures', [])
            if gestures:
                gaze_state['hand_gesture'] = ', '.join(gestures)
            else:
                gaze_state['hand_gesture'] = 'none'
            
            # Detect suspicious behaviors (face touching, chin holding, etc.)
            face_landmarks_for_behavior = None
            if 'detection_result' in locals() and detection_result.face_landmarks:
                face_landmarks_for_behavior = detection_result.face_landmarks[0]
            
            suspicious_behaviors = []
            if hand_details and face_landmarks_for_behavior:
                suspicious_behaviors = detect_suspicious_behaviors(
                    hand_details,
                    face_landmarks_for_behavior,
                    frame.shape
                )
            gaze_state['suspicious_behaviors'] = suspicious_behaviors
            
            # Flag suspicious behaviors
            if suspicious_behaviors:
                behavior_text = ', '.join(suspicious_behaviors).replace('_', ' ').title()
                cv2.putText(frame, f"⚠️ {behavior_text}", (10, frame.shape[0] - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Flag if behavior persists (cooldown to avoid spam)
                if current_time - last_phone_flag_time >= FLAG_COOLDOWN:
                    flag_time = datetime.now().strftime("%H:%M:%S")
                    gaze_state['flags'].append({
                        'time': flag_time,
                        'reason': f'⚠️ Suspicious behavior: {behavior_text}',
                        'duration': '-'
                    })
                    gaze_state['total_flags'] += 1
                    gaze_state['behavior_flags'] += 1
                    last_phone_flag_time = current_time
            
            # Draw detailed hand landmarks with finger tracking
            if hand_landmarks_list:
                h, w = frame.shape[:2]
                for hand_landmarks in hand_landmarks_list:
                    # Draw all 21 landmarks
                    for idx, landmark in enumerate(hand_landmarks):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        
                        # Color code: fingertips are red, joints are green
                        if idx in [4, 8, 12, 16, 20]:  # Fingertips
                            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # Draw connections between landmarks
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                        (5, 9), (9, 13), (13, 17)  # Palm
                    ]
                    
                    for start_idx, end_idx in connections:
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]
                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))
                        cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
            
            # Display hand information on frame
            y_offset = 30
            left_info = hand_details.get('left_hand')
            if left_info:
                fingers = left_info.get('fingers_extended', [])
                finger_names = ['👍', '☝️', '🖕', '💍', '🤙']
                extended_fingers = [finger_names[i] for i, ext in enumerate(fingers) if ext]
                gesture = left_info.get('gesture', 'unknown')
                cv2.putText(frame, f"LEFT: {gesture} {' '.join(extended_fingers)}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
            
            right_info = hand_details.get('right_hand')
            if right_info:
                fingers = right_info.get('fingers_extended', [])
                finger_names = ['👍', '☝️', '🖕', '💍', '🤙']
                extended_fingers = [finger_names[i] for i, ext in enumerate(fingers) if ext]
                gesture = right_info.get('gesture', 'unknown')
                cv2.putText(frame, f"RIGHT: {gesture} {' '.join(extended_fingers)}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
            
            # Flag if holding object
            if holding_object:
                cv2.putText(frame, "🚨 HAND HOLDING OBJECT!", (50, frame.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if current_time - last_phone_flag_time >= FLAG_COOLDOWN:
                    flag_time = datetime.now().strftime("%H:%M:%S")
                    gesture_info = f" ({gaze_state['hand_gesture']})" if gaze_state['hand_gesture'] != 'none' else ''
                    gaze_state['flags'].append({
                        'time': flag_time,
                        'reason': f'✋ Hand holding object!{gesture_info}',
                        'duration': '-'
                    })
                    gaze_state['total_flags'] += 1
                    gaze_state['hand_flags'] += 1
                    last_phone_flag_time = current_time
        
        # Process face with MediaPipe tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = face_landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
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
            # Get first face landmarks (list of NormalizedLandmark objects)
            landmarks = detection_result.face_landmarks[0]
            
            # Get head pose first (needed for gaze direction)
            pitch, yaw, roll = get_head_pose(landmarks, frame.shape)
            
            # Apply calibration offset to pitch
            adjusted_pitch = pitch - gaze_state.get('pitch_offset', 0)
            
            # Get gaze direction using MediaPipe face landmarks
            gaze_data = get_gaze_direction(landmarks, frame.shape, head_yaw=yaw, head_pitch=adjusted_pitch)
            
            # Calculate eye openness and blink detection
            left_openness, right_openness, avg_openness, is_blinking = calculate_eye_openness(landmarks, frame.shape)
            gaze_state['eye_openness_left'] = left_openness
            gaze_state['eye_openness_right'] = right_openness
            gaze_state['eye_openness_avg'] = avg_openness
            gaze_state['is_blinking'] = is_blinking
            
            # Track blinks
            if is_blinking:
                gaze_state['blink_count'] += 1
            
            # Store gaze data in state
            gaze_state['iris_position'] = {
                'left': {'h': round(gaze_data.get('iris_h', 0.5), 2), 'v': round(gaze_data.get('iris_v', 0.5), 2)},
                'right': {'h': round(gaze_data.get('iris_h', 0.5), 2), 'v': round(gaze_data.get('iris_v', 0.5), 2)}
            }
            
            # Store raw values for display
            gaze_state['pitch'] = pitch
            gaze_state['yaw'] = yaw
            gaze_state['head_roll'] = roll
            
            # Draw face mesh with iris tracking
            frame = draw_face_mesh(frame, landmarks, draw_iris=True, draw_contours=False)
            
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
                # Determine if looking at screen using new gaze detection
                is_looking, reason = is_looking_at_screen(gaze_data)
                
                # Calculate full-face attention score
                hand_lm_for_attention = gaze_state.get('hand_landmarks', [])
                attention_data = calculate_attention_score(
                    landmarks, frame.shape, yaw, adjusted_pitch, roll, 
                    hand_landmarks=hand_lm_for_attention
                )
                
                # Update gaze state with attention metrics
                gaze_state['attention_score'] = attention_data['attention_score']
                gaze_state['attention_status'] = attention_data['attention_status']
                gaze_state['facing_score'] = attention_data['facing_score']
                gaze_state['occlusion_detected'] = attention_data['occlusion_detected']
                gaze_state['mouth_activity'] = attention_data['mouth_activity']
                
                # Update state
                gaze_state['status'] = 'tracking'
                gaze_state['direction'] = reason
                gaze_state['is_looking_away'] = not is_looking
                
                # Draw face bounding box
                h, w = frame.shape[:2]
                face_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]]
                x_coords = [p[0] for p in face_points]
                y_coords = [p[1] for p in face_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                color = (0, 255, 0) if is_looking else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Draw eye landmarks
                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    x_pt = int(landmarks[idx].x * w)
                    y_pt = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x_pt, y_pt), 2, (255, 255, 0), -1)
                
                # Draw iris landmarks
                for idx in LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES:
                    x_pt = int(landmarks[idx].x * w)
                    y_pt = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x_pt, y_pt), 2, (0, 255, 255), -1)
                
                # Draw status
                status_text = reason
                status_color = (0, 255, 0) if is_looking else (0, 0, 255)
                cv2.putText(frame, status_text, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Draw head pose info (show adjusted pitch)
                adjusted_pitch = pitch - gaze_state['pitch_offset']
                cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {adjusted_pitch:.1f} (raw: {pitch:.1f})", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw gaze direction indicators
                iris_h = gaze_data.get('iris_h', 0.5)
                iris_v = gaze_data.get('iris_v', 0.5)
                
                # Convert iris position (0-1) to offset from center (-0.5 to 0.5)
                h_offset = iris_h - 0.5
                v_offset = iris_v - 0.5
                
                # Draw directional arrow based on gaze
                center_x, center_y = w // 2, h // 2
                arrow_length = 150
                arrow_color = (0, 255, 255) if is_looking else (0, 0, 255)
                
                # Calculate arrow endpoint based on gaze direction
                end_x = int(center_x + h_offset * arrow_length * 2)
                end_y = int(center_y + v_offset * arrow_length * 2)
                
                # Draw arrow from center
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                               arrow_color, 3, tipLength=0.3)
                
                # Draw gaze values
                direction_text = gaze_data.get('direction', 'center')
                cv2.putText(frame, f"Direction: {direction_text.upper()} | Iris H: {iris_h:.2f} V: {iris_v:.2f}", (50, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw attention score (full-face monitoring)
                att_score = attention_data['attention_score']
                att_status = attention_data['attention_status']
                facing = attention_data['facing_score']
                
                # Color based on attention status
                if att_status == 'attentive':
                    att_color = (0, 255, 0)
                elif att_status == 'distracted':
                    att_color = (0, 165, 255)
                else:  # away or occluded
                    att_color = (0, 0, 255)
                
                cv2.putText(frame, f"Attention: {att_score:.0f}% ({att_status.upper()}) | Facing: {facing:.0f}%", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, att_color, 1)
                
                # Show occlusion warning if detected
                if attention_data['occlusion_detected']:
                    regions = ', '.join(attention_data['occlusion_regions'])
                    cv2.putText(frame, f"OCCLUSION: {regions}", (50, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
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


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

@app.route('/status')
def status():
    """Return current gaze status."""
    return jsonify(convert_to_serializable(gaze_state))


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


@app.route('/tab_switch', methods=['POST'])
def tab_switch():
    """Track tab switching events."""
    data = request.get_json()
    action = data.get('action')
    timestamp = data.get('timestamp')
    
    if action == 'hidden':
        gaze_state['tab_switches'] += 1
        gaze_state['is_tab_visible'] = False
        print(f"⚠️ Tab switched away at {timestamp}")
        
        # Flag immediately when tab is switched
        flag_time = datetime.now().strftime("%H:%M:%S")
        gaze_state['flags'].append({
            'time': flag_time,
            'reason': '🔄 Tab switched away!',
            'duration': '-'
        })
        gaze_state['total_flags'] += 1
        gaze_state['tab_switch_flags'] += 1
    elif action == 'visible':
        gaze_state['is_tab_visible'] = True
        duration = data.get('duration', '0')
        print(f"✅ Tab returned after {duration}s")
    
    return jsonify({'success': True})


@app.route('/window_focus', methods=['POST'])
def window_focus():
    """Track window focus/blur events."""
    data = request.get_json()
    action = data.get('action')
    timestamp = data.get('timestamp')
    
    if action == 'blur':
        gaze_state['window_blurs'] += 1
        print(f"⚠️ Window lost focus at {timestamp}")
        
        # Flag immediately when window loses focus
        flag_time = datetime.now().strftime("%H:%M:%S")
        gaze_state['flags'].append({
            'time': flag_time,
            'reason': '🪟 Window lost focus!',
            'duration': '-'
        })
        gaze_state['total_flags'] += 1
    elif action == 'focus':
        print(f"✅ Window gained focus at {timestamp}")
    
    return jsonify({'success': True})


@app.route('/fullscreen_exit', methods=['POST'])
def fullscreen_exit():
    """Track fullscreen exit events."""
    data = request.get_json()
    timestamp = data.get('timestamp')
    
    flag_time = datetime.now().strftime("%H:%M:%S")
    gaze_state['flags'].append({
        'time': flag_time,
        'reason': '🖥️ Exited fullscreen',
        'duration': '-'
    })
    gaze_state['total_flags'] += 1
    print(f"⚠️ Fullscreen exited at {timestamp}")
    
    return jsonify({'success': True})


@app.route('/window_resize', methods=['POST'])
def window_resize():
    """Track window resize events (catches minimize and window manipulation)."""
    data = request.get_json()
    timestamp = data.get('timestamp')
    old_size = data.get('oldSize', {})
    new_size = data.get('newSize', {})
    
    flag_time = datetime.now().strftime("%H:%M:%S")
    gaze_state['flags'].append({
        'time': flag_time,
        'reason': '↔️ Window resized/moved',
        'duration': '-'
    })
    gaze_state['total_flags'] += 1
    print(f"⚠️ Window resized at {timestamp}: {old_size.get('width')}x{old_size.get('height')} → {new_size.get('width')}x{new_size.get('height')}")
    
    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 50)
    print("Eye Gaze Detection Web App")
    print("=" * 50)
    
    try:
        # Reset all counters to 0 at start
        reset_all_counters()
        
        print("\nInitializing models...")
        models_loaded = init_models()
        print(f"\nModels loaded result: {models_loaded}")
        
        if models_loaded:
            print("\n✅ All models loaded successfully!")
            print("🌐 Starting Flask server at http://localhost:5001")
            print("📹 Open your browser and navigate to http://localhost:5001")
            print("=" * 50)
            print("\nServer is running... Press Ctrl+C to stop\n")
            app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
        else:
            print("\n❌ Failed to load models - server will not start")
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
