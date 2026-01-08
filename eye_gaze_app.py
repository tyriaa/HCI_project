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
    'behavior_flags': 0
}

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
    
    # Suppress MediaPipe warnings
    import warnings
    warnings.filterwarnings('ignore')
    
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
        # Small delay to let it fully initialize
        time.sleep(0.1)
        print("✅ MediaPipe Face Landmarker loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Face Landmarker: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Initialize MediaPipe Hand Landmarker
        print("Loading MediaPipe Hand Landmarker...")
        hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        time.sleep(0.1)
        print("✅ MediaPipe Hand Landmarker loaded")
    except Exception as e:
        print(f"⚠️ Warning: Hand Landmarker failed to load: {e}")
        print("⚠️ Continuing without hand detection...")
        hand_landmarker = None
    
    try:
        # Load YOLO model for phone detection
        print("Loading YOLO model for phone detection...")
        yolo_model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All models initialized successfully!")
    print("=" * 50)
    return True


def detect_phone(frame):
    """Detect cell phone in frame using YOLO."""
    if yolo_model is None:
        return False, None
    
    phone_detected = False
    phone_box = None
    
    # Run YOLO detection
    results = yolo_model(frame, verbose=False, conf=PHONE_CONFIDENCE_THRESHOLD)
    
    for result in results:
        for box in result.boxes:
            # Check if detected object is a cell phone
            if int(box.cls[0]) == PHONE_CLASS_ID:
                phone_detected = True
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                phone_box = (x1, y1, x2, y2, conf)
                break
    
    return phone_detected, phone_box


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
        
        # Detection Method 1: Closed fist or gripping gesture
        if gesture in ['closed_fist', 'gripping']:
            holding_object = True
        
        # Detection Method 2: Hand near detected phone
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
        
        hand_x, hand_y = hand_info['position']
        gesture = hand_info['gesture']
        
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


# MediaPipe landmark indices for eyes
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


def get_eye_region(landmarks, eye_indices, img_shape):
    """Extract eye region coordinates from MediaPipe landmarks."""
    h, w = img_shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_indices])
    return pts


def get_gaze_ratio(eye_indices, iris_indices, gray, landmarks, img_shape):
    """Calculate gaze ratio using MediaPipe iris landmarks with horizontal and vertical tracking."""
    h, w = img_shape[:2]
    
    # Get eye region
    eye_region = get_eye_region(landmarks, eye_indices, img_shape)
    
    # Get bounding box
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    
    # Add padding
    padding = 5
    min_x = max(0, min_x - padding)
    max_x = min(w, max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(h, max_y + padding)
    
    eye = gray[min_y:max_y, min_x:max_x]
    
    if eye.size == 0:
        return (1.0, 0.0), "unknown"
    
    # Get iris center (both x and y)
    iris_x = int(landmarks[iris_indices[0]].x * w)
    iris_y = int(landmarks[iris_indices[0]].y * h)
    
    eye_center_x = (min_x + max_x) // 2
    eye_center_y = (min_y + max_y) // 2
    
    eye_width = max_x - min_x
    eye_height = max_y - min_y
    
    if eye_width == 0 or eye_height == 0:
        return (1.0, 0.0), "center"
    
    # Calculate relative position
    # Horizontal: -1 = far left, 0 = center, 1 = far right
    # Vertical: positive = looking down (iris moves down in image), negative = looking up (iris moves up in image)
    horizontal_pos = (iris_x - eye_center_x) / (eye_width / 2)
    vertical_pos = (iris_y - eye_center_y) / (eye_height / 2)
    
    # Determine direction with tighter thresholds
    h_threshold = 0.12
    v_threshold = 0.15
    
    if abs(vertical_pos) > v_threshold:
        if vertical_pos > v_threshold:
            direction = "down"  # Positive vertical_pos = iris moved down = looking down
        else:
            direction = "up"  # Negative vertical_pos = iris moved up = looking up
    elif abs(horizontal_pos) > h_threshold:
        if horizontal_pos < -h_threshold:
            direction = "left"
        else:
            direction = "right"
    else:
        direction = "center"
    
    return (horizontal_pos, vertical_pos), direction


def get_head_pose(landmarks, img_shape):
    """Estimate head pose using MediaPipe landmarks."""
    h, w = img_shape[:2]
    
    # MediaPipe face landmark indices
    # 1: Nose tip, 152: Chin, 33: Left eye, 263: Right eye, 61: Left mouth, 291: Right mouth
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),  # Chin
        (landmarks[33].x * w, landmarks[33].y * h),    # Left eye left corner
        (landmarks[263].x * w, landmarks[263].y * h),  # Right eye right corner
        (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth corner
        (landmarks[291].x * w, landmarks[291].y * h)   # Right mouth corner
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
    focal_length = w
    center = (w / 2, h / 2)
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


def is_looking_at_screen(left_gaze, right_gaze, left_dir, right_dir, pitch, yaw, pitch_offset=0):
    """Determine if user is looking at the screen with improved directional detection."""
    # Apply calibration offset to pitch
    adjusted_pitch = pitch - pitch_offset
    
    # Adjusted thresholds for better accuracy
    YAW_THRESHOLD = 25  # Head turn left/right
    PITCH_UP_THRESHOLD = 30  # Looking up (more lenient for top of screen)
    PITCH_DOWN_THRESHOLD = 25  # Looking down
    
    # Extract horizontal and vertical gaze positions
    left_h, left_v = left_gaze
    right_h, right_v = right_gaze
    avg_h = (left_h + right_h) / 2
    avg_v = (left_v + right_v) / 2
    
    # Priority 1: Check head pose (most reliable for large movements)
    if abs(yaw) > YAW_THRESHOLD:
        if yaw > 0:
            return False, "👉 Head turned right"
        else:
            return False, "👈 Head turned left"
    
    if adjusted_pitch > PITCH_DOWN_THRESHOLD:
        return False, "👇 Head looking down"
    elif adjusted_pitch < -PITCH_UP_THRESHOLD:
        return False, "👆 Head looking up"
    
    # Priority 2: Check eye gaze for subtle movements
    # Vertical eye movement (positive = down, negative = up)
    if avg_v > 0.2:
        return False, "👇 Eyes looking down"
    elif avg_v < -0.2:
        return False, "👆 Eyes looking up"
    
    # Horizontal eye movement
    if left_dir == "left" and right_dir == "left":
        return False, "👈 Eyes looking left"
    elif left_dir == "right" and right_dir == "right":
        return False, "👉 Eyes looking right"
    
    # Combined check: if eyes and head slightly disagree
    if avg_h < -0.15:
        return False, "👈 Looking left"
    elif avg_h > 0.15:
        return False, "👉 Looking right"
    
    return True, "✅ Looking at screen"


# Tracking variables
looking_away_start = None
last_flag_time = 0


def generate_frames():
    """Generate video frames with gaze detection overlay."""
    global gaze_state, looking_away_start, last_flag_time, calibration_samples, last_phone_flag_time
    
    if face_landmarker is None:
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
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
            gaze_state['left_hand_visible'] = hand_details['left_hand'] is not None
            gaze_state['right_hand_visible'] = hand_details['right_hand'] is not None
            
            # Update finger states
            if hand_details['left_hand']:
                gaze_state['fingers_extended']['left'] = hand_details['left_hand']['fingers_extended']
            else:
                gaze_state['fingers_extended']['left'] = []
                
            if hand_details['right_hand']:
                gaze_state['fingers_extended']['right'] = hand_details['right_hand']['fingers_extended']
            else:
                gaze_state['fingers_extended']['right'] = []
            
            # Update gesture
            if hand_details['gestures']:
                gaze_state['hand_gesture'] = ', '.join(hand_details['gestures'])
            else:
                gaze_state['hand_gesture'] = 'none'
            
            # Detect suspicious behaviors (face touching, chin holding, etc.)
            face_landmarks_for_behavior = None
            if 'detection_result' in locals() and detection_result.face_landmarks:
                face_landmarks_for_behavior = detection_result.face_landmarks[0]
            
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
            if hand_details['left_hand']:
                left_info = hand_details['left_hand']
                fingers = left_info['fingers_extended']
                finger_names = ['👍', '☝️', '🖕', '💍', '🤙']
                extended_fingers = [finger_names[i] for i, ext in enumerate(fingers) if ext]
                cv2.putText(frame, f"LEFT: {left_info['gesture']} {' '.join(extended_fingers)}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
            
            if hand_details['right_hand']:
                right_info = hand_details['right_hand']
                fingers = right_info['fingers_extended']
                finger_names = ['👍', '☝️', '🖕', '💍', '🤙']
                extended_fingers = [finger_names[i] for i, ext in enumerate(fingers) if ext]
                cv2.putText(frame, f"RIGHT: {right_info['gesture']} {' '.join(extended_fingers)}", 
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
            
            # Get eye gaze using iris landmarks (now returns position tuple and direction)
            left_gaze, left_dir = get_gaze_ratio(LEFT_EYE_INDICES, LEFT_IRIS_INDICES, gray, landmarks, frame.shape)
            right_gaze, right_dir = get_gaze_ratio(RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES, gray, landmarks, frame.shape)
            
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
                # Determine if looking at screen (with calibration offset and improved detection)
                is_looking, reason = is_looking_at_screen(
                    left_gaze, right_gaze, left_dir, right_dir, pitch, yaw, gaze_state['pitch_offset']
                )
                
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
                left_h, left_v = left_gaze
                right_h, right_v = right_gaze
                avg_h = (left_h + right_h) / 2
                avg_v = (left_v + right_v) / 2
                
                # Draw directional arrow based on gaze
                center_x, center_y = w // 2, h // 2
                arrow_length = 80
                arrow_color = (0, 255, 255) if is_looking else (0, 0, 255)
                
                # Calculate arrow endpoint based on gaze direction
                end_x = int(center_x + avg_h * arrow_length)
                end_y = int(center_y + avg_v * arrow_length)
                
                # Draw arrow from center
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                               arrow_color, 3, tipLength=0.3)
                
                # Draw gaze values
                cv2.putText(frame, f"Gaze H: {avg_h:.2f} V: {avg_v:.2f}", (50, 120),
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
