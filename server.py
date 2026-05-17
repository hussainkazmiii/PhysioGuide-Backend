import cv2
import mediapipe as mp
import numpy as np
import zmq
import json
import time
import sys
import os
import socket
import threading
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
from reference_profile import (
    ReferenceProfile, load_profile, get_default_profile
)

def normalize_exercise_name(name):
    return name.lower().strip()

def perfect_rep_tips(exercise_type):
    return []

# ==========================================
# CONFIGURATION
# ==========================================
PORT = "5555"
CONTROL_PORT = "5557"  # 🔥 NEW (Unity control)

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(base_path, "reference_profiles")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(base_path, "pose_landmarker_lite.task")
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"[System] Downloading {MODEL_PATH} (this might take a minute)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[System] Download complete.")

# ==========================================
# PART 1: VECTOR MATH & BIOMECHANICS ENGINE
# ==========================================

def calculate_angle_3d(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ==========================================
# PART 2: PRECAUTION MESSAGE GENERATOR
# ==========================================

PRECAUTION_MESSAGES = {
    "shoulder_abduction": {
        "hiking": "Lower your shoulders — don't shrug while lifting",
        "bent_elbow": "Straighten your elbows — keep arms extended",
        "SEA_low": "Raise your arms higher to match the reference form",
        "SEA_high": "Don't raise arms too high — stay within the target range",
    },
    "chin_tuck": {
        "hiking": "Keep your shoulders relaxed and down",
        "flexion": "Don't tilt your chin down — keep your head level",
        "extension": "Don't tilt your chin up — keep your head level",
        "HRM_low": "Tuck your chin further back toward the reference range",
        "HRM_high": "Don't retract too far — stay within comfortable range",
    },
}
PRECAUTION_MESSAGES["scaption"] = PRECAUTION_MESSAGES["shoulder_abduction"]

def generate_precautions(exercise_type, error_flags, deviations, profile):
    exercise_type = normalize_exercise_name(exercise_type)
    precautions = []
    msgs = PRECAUTION_MESSAGES.get(exercise_type, {})

    for error_name, is_active in error_flags.items():
        if is_active and error_name in msgs:
            precautions.append(msgs[error_name])

    if profile and profile.video_source and profile.video_source != "none (using hardcoded defaults)":
        for metric, dev_val in deviations.items():
            low_key = f"{metric}_low"
            high_key = f"{metric}_high"
            if dev_val < -5 and low_key in msgs:
                precautions.append(f"{msgs[low_key]} ({abs(dev_val):.1f}° below reference)")
            elif dev_val > 5 and high_key in msgs:
                precautions.append(f"{msgs[high_key]} ({abs(dev_val):.1f}° above reference)")

    return precautions


# ==========================================
# PART 3: EXERCISE PROCESSOR
# ==========================================

class ExerciseProcessor:
    def __init__(self):
        self.detector = None # Initialize lazily when starting
        
        self.state = "NEUTRAL"
        self.rep_count = 0
        self.hold_timer_start = 0
        
        self.baseline_shoulder_y = None
        self.baseline_nose_z = None 
        self.calibrated = False
        self.start_time = time.time() 

        self.profiles = {}
        self.lstm_classifier = None
        print("[System] LSTM classifier temporarily disabled for cloud deployment.")

    def load_model(self):
        if self.detector is None:
            download_model_if_needed()
            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("[System] MediaPipe Tasks ML Model Loaded.")

    def unload_model(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
            print("[System] MediaPipe Tasks ML Model Unloaded.")

    def load_profiles(self):
        for exercise in ["shoulder_abduction", "chin_tuck"]:
            profile_path = os.path.join(PROFILES_DIR, f"{exercise}.json")
            profile = load_profile(profile_path)
            if profile:
                self.profiles[exercise] = profile
                print(f"[System] ✓ Loaded TRAINED profile for '{exercise}' "
                      f"({profile.total_reps} reps, tolerance={profile.tolerance_multiplier}x)")
            else:
                self.profiles[exercise] = get_default_profile(exercise)
                print(f"[System] ⚠ No trained profile for '{exercise}' — using hardcoded defaults")

        self.profiles["scaption"] = self.profiles.get("shoulder_abduction")

    def get_profile(self, exercise_type):
        return self.profiles.get(normalize_exercise_name(exercise_type))

    def get_coords(self, landmarks, idx):
        return [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

    def _get_transition(self, profile, name, key, fallback):
        if profile:
            trans = profile.get_state_transition(name)
            if trans and isinstance(trans, dict) and key in trans:
                return trans[key]
        return fallback

    def _get_hold_duration(self, profile, fallback):
        if profile:
            val = profile.get_state_transition("HOLDING_duration")
            if val is not None and not isinstance(val, dict):
                return val
        return fallback

    def process_frame(self, image, exercise_type="shoulder_abduction"):
        exercise_type = normalize_exercise_name(exercise_type)
        if self.detector is None:
            return None, None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        
        if not results.pose_world_landmarks or len(results.pose_world_landmarks) == 0:
            return None, None

        w_lm = results.pose_world_landmarks[0]
        profile = self.get_profile(exercise_type)
        
        avg_shoulder_y = (w_lm[11].y + w_lm[12].y) / 2
        shm_val = 0.0 

        if not self.calibrated:
            if (time.time() - self.start_time) > 3.0:
                self.baseline_shoulder_y = avg_shoulder_y
                self.baseline_nose_z = w_lm[0].z 
                self.calibrated = True
                print("[System] Baseline Calibrated. POSTURE LOCKED.")
            else:
                shm_val = 0.0
        else:
            shm_val = abs(self.baseline_shoulder_y - avg_shoulder_y)

        metrics = {}
        error_flags = {}
        deviations = {}
        
        if exercise_type == "shoulder_abduction":
            l_sea = calculate_angle_3d(self.get_coords(w_lm, 23), self.get_coords(w_lm, 11), self.get_coords(w_lm, 13))
            r_sea = calculate_angle_3d(self.get_coords(w_lm, 24), self.get_coords(w_lm, 12), self.get_coords(w_lm, 14))
            avg_sea = (l_sea + r_sea) / 2
            
            l_ea = calculate_angle_3d(self.get_coords(w_lm, 11), self.get_coords(w_lm, 13), self.get_coords(w_lm, 15))
            r_ea = calculate_angle_3d(self.get_coords(w_lm, 12), self.get_coords(w_lm, 14), self.get_coords(w_lm, 16))
            avg_ea = (l_ea + r_ea) / 2

            metrics = {"SEA": round(avg_sea, 1), "EA": round(avg_ea, 1), "SHM": round(shm_val, 3)}

            hiking_max = profile.get_error_max("hiking") if profile else 0.05
            elbow_min = profile.get_error_min("bent_elbow") if profile else 140
            error_flags["hiking"] = bool(shm_val > (hiking_max or 0.05))
            error_flags["bent_elbow"] = bool(avg_ea < (elbow_min or 140))

            if profile:
                ref_sea = profile.get_reference_value("SEA", self.state)
                ref_ea = profile.get_reference_value("EA", self.state)
                if ref_sea is not None:
                    deviations["SEA"] = round(avg_sea - ref_sea, 1)
                if ref_ea is not None:
                    deviations["EA"] = round(avg_ea - ref_ea, 1)

            t_neutral_lift = self._get_transition(profile, "NEUTRAL_to_LIFTING", "SEA_threshold", 30)
            t_lift_hold = self._get_transition(profile, "LIFTING_to_HOLDING", "SEA_threshold", 85)
            t_lower_neutral = self._get_transition(profile, "LOWERING_to_NEUTRAL", "SEA_threshold", 30)
            t_lift_fallback = self._get_transition(profile, "LIFTING_fallback", "SEA_threshold", 25)
            t_hold_fallback = self._get_transition(profile, "HOLDING_fallback", "SEA_threshold", 70)
            t_hold_duration = self._get_hold_duration(profile, 0.5)

            if self.state == "NEUTRAL":
                if avg_sea > t_neutral_lift: self.state = "LIFTING"
            elif self.state == "LIFTING":
                if avg_sea > t_lift_hold:
                    self.state = "HOLDING"
                    self.hold_timer_start = time.time()
                elif avg_sea < t_lift_fallback: self.state = "NEUTRAL"
            elif self.state == "HOLDING":
                if (time.time() - self.hold_timer_start) > t_hold_duration: self.state = "LOWERING"
                elif avg_sea < t_hold_fallback: self.state = "LIFTING"
            elif self.state == "LOWERING":
                if avg_sea < t_lower_neutral:
                    self.state = "NEUTRAL"
                    self.rep_count += 1
                    print(f"[Gameplay] Shoulder Abduction Rep Completed! Total: {self.rep_count}")

        elif exercise_type == "chin_tuck":
            current_nose_z = w_lm[0].z
            hrm_val = abs((current_nose_z - self.baseline_nose_z) * 100) if self.baseline_nose_z else 0.0
            
            ear_y = (w_lm[7].y + w_lm[8].y) / 2
            nam_diff = w_lm[0].y - ear_y 

            metrics = {"HRM": round(hrm_val, 2), "NAM": round(nam_diff, 3), "SMI": round(shm_val, 3)}

            hiking_max = profile.get_error_max("hiking") if profile else 0.03
            flexion_max = profile.get_error_max("flexion") if profile else 0.05
            extension_min = profile.get_error_min("extension") if profile else -0.05
            error_flags["hiking"] = bool(shm_val > (hiking_max or 0.03))
            error_flags["flexion"] = bool(nam_diff > (flexion_max or 0.05))
            error_flags["extension"] = bool(nam_diff < (extension_min or -0.05))

            if profile:
                ref_hrm = profile.get_reference_value("HRM", self.state)
                ref_nam = profile.get_reference_value("NAM", self.state)
                if ref_hrm is not None:
                    deviations["HRM"] = round(hrm_val - ref_hrm, 2)
                if ref_nam is not None:
                    deviations["NAM"] = round(nam_diff - ref_nam, 3)

            t_neutral_retract = self._get_transition(profile, "NEUTRAL_to_RETRACTING", "HRM_threshold", 1.0)
            t_retract_hold = self._get_transition(profile, "RETRACTING_to_HOLDING", "HRM_threshold", 2.0)
            t_return_neutral = self._get_transition(profile, "RETURNING_to_NEUTRAL", "HRM_threshold", 0.5)
            t_retract_fallback = self._get_transition(profile, "RETRACTING_fallback", "HRM_threshold", 0.5)
            t_hold_fallback = self._get_transition(profile, "HOLDING_fallback", "HRM_threshold", 1.5)
            t_hold_duration = self._get_hold_duration(profile, 1.0)

            if self.state == "NEUTRAL":
                if hrm_val > t_neutral_retract: self.state = "RETRACTING"
            elif self.state == "RETRACTING":
                if hrm_val > t_retract_hold: 
                    self.state = "HOLDING"
                    self.hold_timer_start = time.time()
                elif hrm_val < t_retract_fallback: self.state = "NEUTRAL"
            elif self.state == "HOLDING":
                if (time.time() - self.hold_timer_start) > t_hold_duration: self.state = "RETURNING"
                elif hrm_val < t_hold_fallback: self.state = "RETRACTING"
            elif self.state == "RETURNING":
                if hrm_val < t_return_neutral: 
                    self.state = "NEUTRAL"
                    self.rep_count += 1
                    print(f"[Gameplay] Chin Tuck Rep Completed! Total: {self.rep_count}")

        precautions = generate_precautions(exercise_type, error_flags, deviations, profile)

        raw_list = []
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            for lm in results.pose_landmarks[0]:
                raw_list.append({"x": -lm.x, "y": lm.y, "z": lm.z, "vis": lm.visibility})

        model_feedback = {
        "enabled": False,
        "ready": False}

        json_packet = {
            "timestamp": int(time.time() * 1000),
            "exercise": exercise_type,
            "metrics": metrics,
            "error_flags": error_flags,
            "deviations": deviations,
            "precautions": precautions,
            "model_feedback": model_feedback,
            "state": self.state,
            "rep_count": self.rep_count,
            "raw_landmarks": raw_list,
            "status": "tracking",
            "profile_source": profile.video_source if profile else "none",
        }
        
        return json_packet, results

# ==========================================
# MAIN SERVER LOOP
# ==========================================

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def udp_broadcast_worker():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    msg = json.dumps({"service": "PhysioGuide"}).encode('utf-8')
    while True:
        try:
            # Broadcast to 255.255.255.255 on port 5559
            s.sendto(msg, ("255.255.255.255", 5559))
        except Exception:
            pass
        time.sleep(1.0)

def run_server():
    context = zmq.Context()
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind(f"tcp://0.0.0.0:{PORT}")

    # 🔥 NEW CONTROL SOCKET
    control_socket = context.socket(zmq.REP)
    control_socket.bind(f"tcp://0.0.0.0:{CONTROL_PORT}")
    control_socket.RCVTIMEO = 10  # Very short timeout so we don't block video loop

    local_ip = get_local_ip()
    print(f"=========================================")
    print(f"    >>>> YOUR LOCAL IP IS: {local_ip} <<<<")
    print(f"=========================================")
    print(f"[Network] Server broadcast ready on TCP Port {PORT} (All Interfaces)")
    print(f"[Network] Server listening for control commands on TCP Port {CONTROL_PORT}")
    print(f"=========================================")
    
    discovery_thread = threading.Thread(target=udp_broadcast_worker, daemon=True)
    discovery_thread.start()
    
        
    processor = ExerciseProcessor()
    frame_send_count = 0
    processor.load_profiles()
    
    current_exercise = "shoulder_abduction" 
    is_recording = False
    recorded_frames = []

    try:
        while True:

            # 🔥 UNITY COMMAND HANDLER
            try:
                msg = control_socket.recv_string()
                print(f"[Network] >> Received command from Unity: {msg}")
                
                try:
                    command = json.loads(msg)
                    cmd_action = command.get("command")

                    if cmd_action == "HELLO":
                        print(f"[Network] \n>>>> UNITY CLIENT ATTACHED AND HANDSHAKE COMPLETED <<<<\n")

                    elif cmd_action == "START":
                        current_exercise = normalize_exercise_name(
                            command.get("exercise", "shoulder_abduction")
                        )
                        processor.rep_count = 0
                        processor.state = "NEUTRAL"
                        processor.calibrated = False
                        processor.start_time = time.time()
                        recorded_frames = []
                        is_recording = True
                        processor.load_model()
                        print(f"[Unity] START - State: TRACKING, Exercise: {current_exercise}")

                    elif cmd_action == "STOP":
                        is_recording = False
                        processor.unload_model()
                        print("[Unity] STOP - State: WAITING_FOR_UNITY")
                    
                    # IMPORTANT: Must reply because this is a REP socket!
                    control_socket.send_string("OK")
                    
                except json.JSONDecodeError:
                    print(f"[Error] Invalid JSON received from Unity: {repr(msg)}")
                    control_socket.send_string("ERROR: Invalid JSON")
                except Exception as e:
                    print(f"[Error] Unexpected error parsing command: {e}")
                    control_socket.send_string(f"ERROR: {str(e)}")

            except zmq.Again:
                pass

            json_packet, results = None, None

            if is_recording:
                try:
                    json_packet, results = None, None
                    # json_packet, results = processor.process_frame(frame, exercise_type=current_exercise)
                    if results is None:
                        # Print occasionally to avoid spamming the console
                        if int(time.time() * 10) % 30 == 0:
                            print("[System] No pose detected by MediaPipe. Waiting for subject...")
                except Exception as e:
                    import traceback
                    print(f"[Error] Exception in process_frame: {e}")
                    traceback.print_exc()

            if json_packet and results:
                socket_pub.send_string(json.dumps(json_packet))
                frame_send_count += 1
                if frame_send_count % 30 == 0:
                    print(f"[Network] Successfully broadcasted frame {frame_send_count} to Unity.")

                # if results.pose_landmarks and len(results.pose_landmarks) > 0:
                #     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                #     pose_landmarks_proto.landmark.extend([
                #         landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility) 
                #         for landmark in results.pose_landmarks[0]
                #     ])
                #     mp.solutions.drawing_utils.draw_landmarks(
                #         frame, pose_landmarks_proto, mp.solutions.pose.POSE_CONNECTIONS)

            

    finally:
        socket_pub.close()
        control_socket.close()
        context.term()

if __name__ == "__main__":
    run_server()
