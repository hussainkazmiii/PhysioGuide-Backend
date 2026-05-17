import cv2
import mediapipe as mp
import numpy as np
import zmq
import json
import time
import sys
import os

from reference_profile import (
    ReferenceProfile, load_profile, get_default_profile
)


# ==========================================
# CONFIGURATION
# ==========================================
PORT = "5555"
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_profiles")

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
    "scaption": {
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

def generate_precautions(exercise_type, error_flags, deviations, profile):
    """
    Generate human-readable precaution messages based on current errors
    and deviations from the reference profile.
    """
    precautions = []
    msgs = PRECAUTION_MESSAGES.get(exercise_type, {})

    # Error-based precautions
    for error_name, is_active in error_flags.items():
        if is_active and error_name in msgs:
            precautions.append(msgs[error_name])

    # Deviation-based precautions (only if we have a trained profile)
    if profile and profile.video_source and profile.video_source != "none (using hardcoded defaults)":
        for metric, dev_val in deviations.items():
            low_key = f"{metric}_low"
            high_key = f"{metric}_high"
            # Only flag significant deviations (> 1 std worth)
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
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.state = "NEUTRAL"
        self.rep_count = 0
        self.hold_timer_start = 0
        
        # Calibration
        self.baseline_shoulder_y = None
        self.baseline_nose_z = None 
        self.calibrated = False
        self.start_time = time.time() 

        # Reference Profiles (loaded externally)
        self.profiles = {}

    def load_profiles(self):
        """Load reference profiles from disk, falling back to defaults."""
        for exercise in ["scaption", "chin_tuck"]:
            profile_path = os.path.join(PROFILES_DIR, f"{exercise}.json")
            profile = load_profile(profile_path)
            if profile:
                self.profiles[exercise] = profile
                print(f"[System] ✓ Loaded TRAINED profile for '{exercise}' "
                      f"({profile.total_reps} reps, tolerance={profile.tolerance_multiplier}x)")
            else:
                self.profiles[exercise] = get_default_profile(exercise)
                print(f"[System] ⚠ No trained profile for '{exercise}' — using hardcoded defaults")

    def get_profile(self, exercise_type):
        """Get the active profile for an exercise."""
        return self.profiles.get(exercise_type)

    def get_coords(self, landmarks, idx):
        return [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

    def _get_transition(self, profile, name, key, fallback):
        """Safely get a state transition threshold value."""
        if profile:
            trans = profile.get_state_transition(name)
            if trans and isinstance(trans, dict) and key in trans:
                return trans[key]
        return fallback

    def _get_hold_duration(self, profile, fallback):
        """Get the hold duration from profile."""
        if profile:
            val = profile.get_state_transition("HOLDING_duration")
            if val is not None and not isinstance(val, dict):
                return val
        return fallback

    def process_frame(self, image, exercise_type="scaption"):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_world_landmarks:
            return None, None

        w_lm = results.pose_world_landmarks.landmark
        profile = self.get_profile(exercise_type)
        
        # Metrics
        avg_shoulder_y = (w_lm[11].y + w_lm[12].y) / 2
        shm_val = 0.0 

        # Calibration
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
        
        # =====================================================
        # 1. SCAPTION LOGIC (with profile-based thresholds)
        # =====================================================
        if exercise_type == "scaption":
            l_sea = calculate_angle_3d(self.get_coords(w_lm, 23), self.get_coords(w_lm, 11), self.get_coords(w_lm, 13))
            r_sea = calculate_angle_3d(self.get_coords(w_lm, 24), self.get_coords(w_lm, 12), self.get_coords(w_lm, 14))
            avg_sea = (l_sea + r_sea) / 2
            
            l_ea = calculate_angle_3d(self.get_coords(w_lm, 11), self.get_coords(w_lm, 13), self.get_coords(w_lm, 15))
            r_ea = calculate_angle_3d(self.get_coords(w_lm, 12), self.get_coords(w_lm, 14), self.get_coords(w_lm, 16))
            avg_ea = (l_ea + r_ea) / 2

            metrics = {"SEA": round(avg_sea, 1), "EA": round(avg_ea, 1), "SHM": round(shm_val, 3)}

            # --- Error flags from profile thresholds ---
            hiking_max = profile.get_error_max("hiking") if profile else 0.05
            elbow_min = profile.get_error_min("bent_elbow") if profile else 140
            error_flags["hiking"] = bool(shm_val > (hiking_max or 0.05))
            error_flags["bent_elbow"] = bool(avg_ea < (elbow_min or 140))

            # --- Deviations from reference ---
            if profile:
                ref_sea = profile.get_reference_value("SEA", self.state)
                ref_ea = profile.get_reference_value("EA", self.state)
                if ref_sea is not None:
                    deviations["SEA"] = round(avg_sea - ref_sea, 1)
                if ref_ea is not None:
                    deviations["EA"] = round(avg_ea - ref_ea, 1)

            # --- State Machine (profile-based transitions) ---
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
                    print(f"[Gameplay] Scaption Rep Completed! Total: {self.rep_count}")

        # =====================================================
        # 2. CHIN TUCK LOGIC (with profile-based thresholds)
        # =====================================================
        elif exercise_type == "chin_tuck":
            current_nose_z = w_lm[0].z
            hrm_val = abs((current_nose_z - self.baseline_nose_z) * 100) if self.baseline_nose_z else 0.0
            
            ear_y = (w_lm[7].y + w_lm[8].y) / 2
            nam_diff = w_lm[0].y - ear_y 

            metrics = {"HRM": round(hrm_val, 2), "NAM": round(nam_diff, 3), "SMI": round(shm_val, 3)}

            # --- Error flags from profile thresholds ---
            hiking_max = profile.get_error_max("hiking") if profile else 0.03
            flexion_max = profile.get_error_max("flexion") if profile else 0.05
            extension_min = profile.get_error_min("extension") if profile else -0.05
            error_flags["hiking"] = bool(shm_val > (hiking_max or 0.03))
            error_flags["flexion"] = bool(nam_diff > (flexion_max or 0.05))
            error_flags["extension"] = bool(nam_diff < (extension_min or -0.05))

            # --- Deviations from reference ---
            if profile:
                ref_hrm = profile.get_reference_value("HRM", self.state)
                ref_nam = profile.get_reference_value("NAM", self.state)
                if ref_hrm is not None:
                    deviations["HRM"] = round(hrm_val - ref_hrm, 2)
                if ref_nam is not None:
                    deviations["NAM"] = round(nam_diff - ref_nam, 3)

            # --- State Machine (profile-based transitions) ---
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

        # --- Generate Precautions ---
        precautions = generate_precautions(exercise_type, error_flags, deviations, profile)

        # --- JSON Pack ---
        raw_list = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                raw_list.append({"x": -lm.x, "y": lm.y, "z": lm.z, "vis": lm.visibility})

        json_packet = {
            "timestamp": int(time.time() * 1000),
            "exercise": exercise_type,
            "metrics": metrics,
            "error_flags": error_flags,
            "deviations": deviations,
            "precautions": precautions,
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

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # [BIND TO ALL INTERFACES]
    socket.bind(f"tcp://0.0.0.0:{PORT}")
    print(f"[Network] Server broadcasting on TCP Port {PORT} (All Interfaces)")
    
    cap = cv2.VideoCapture(0)
    processor = ExerciseProcessor()
    
    # LOAD REFERENCE PROFILES
    processor.load_profiles()
    
    current_exercise = "scaption" 
    
    # --- RECORDING VARIABLES ---
    is_recording = False
    recorded_frames = [] # Buffer to store the list of frames
    
    print("\n[System] Vision Engine Started.")
    print("=" * 60)
    print(" CONTROLS:")
    print("   [1] Scaption Mode   [2] Chin Tuck Mode")
    print("   [SPACE] Start/Stop Recording")
    print("   [S] Save Recording  [R] Reset Calibration")
    print("   [Q] Quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            json_packet, results = processor.process_frame(frame, exercise_type=current_exercise)
            
            if json_packet and results:
                # 1. NETWORK: Send data
                socket.send_string(json.dumps(json_packet))

                # 2. RECORDING LOGIC
                if is_recording:
                    recorded_frames.append(json_packet)
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, f"REC: {len(recorded_frames)}", (50, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Visuals (Skeleton)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                
                # --- VISUALS: TEXT OVERLAY ---
                
                # 1. MODE & STATE
                cv2.putText(frame, f"Mode: {current_exercise.upper()}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"State: {json_packet['state']}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 2. REP COUNT
                cv2.putText(frame, f"Reps: {json_packet['rep_count']}", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # 3. PROFILE SOURCE INDICATOR
                src = json_packet.get("profile_source", "none")
                if src and src != "none (using hardcoded defaults)":
                    cv2.putText(frame, f"Profile: TRAINED", (10, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, f"Profile: DEFAULT", (10, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                # 4. ERROR FLAGS
                y_offset = 200
                for err, active in json_packet['error_flags'].items():
                    if active:
                        cv2.putText(frame, f"ERR: {err.upper()}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        y_offset += 30

                # 5. PRECAUTIONS (new!)
                for precaution in json_packet.get('precautions', []):
                    cv2.putText(frame, precaution[:50], (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
                    y_offset += 25

                # 6. DEBUG METRICS
                if current_exercise == "scaption":
                    sea_val = json_packet['metrics'].get('SEA', 0)
                    cv2.putText(frame, f"SEA: {sea_val} deg", (400, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    # Show deviation if available
                    sea_dev = json_packet.get('deviations', {}).get('SEA')
                    if sea_dev is not None:
                        dev_color = (0, 255, 0) if abs(sea_dev) < 10 else (0, 0, 255)
                        cv2.putText(frame, f"Dev: {sea_dev:+.1f} deg", (400, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dev_color, 2)
                elif current_exercise == "chin_tuck":
                    hrm_val = json_packet['metrics'].get('HRM', 0)
                    cv2.putText(frame, f"HRM: {hrm_val} cm", (400, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    hrm_dev = json_packet.get('deviations', {}).get('HRM')
                    if hrm_dev is not None:
                        dev_color = (0, 255, 0) if abs(hrm_dev) < 1 else (0, 0, 255)
                        cv2.putText(frame, f"Dev: {hrm_dev:+.2f} cm", (400, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dev_color, 2)

            # Controls
            cv2.imshow('PhysioGuide Backend', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): break
            elif key == ord('r'): 
                processor.rep_count = 0
                processor.state = "NEUTRAL"
                processor.calibrated = False 
                processor.start_time = time.time()
                recorded_frames = []
                is_recording = False
                print("[System] Resetting Calibration...")

            # --- RECORDING CONTROLS ---
            elif key == 32: # SPACEBAR
                if not is_recording:
                    is_recording = True
                    recorded_frames = [] 
                    print(f"[System] Recording STARTED for {current_exercise}...")
                else:
                    is_recording = False
                    print(f"[System] Recording STOPPED. Captured {len(recorded_frames)} frames.")

            elif key == ord('s'):  # SAVE
                if len(recorded_frames) > 0:
                    filename = f"recording_{current_exercise}.json"

                    # Save locally (full data)
                    with open(filename, 'w') as f:
                        json.dump(recorded_frames, f, indent=4)

                    print(f"[System] SAVED locally to {filename}")

                    # 🔥 CLEAN DATA (IMPORTANT)
                    try:
                        clean_frames = []

                        for f in recorded_frames:
                            clean_frames.append({
                                "timestamp": f["timestamp"],
                                "metrics": f["metrics"],
                                "state": f["state"],
                                "rep_count": f["rep_count"]
                            })

                        # 🔥 LIMIT SIZE (VERY IMPORTANT)
                        clean_frames = clean_frames[:100]  # prevent overflow

            elif key == ord('1'):
                current_exercise = "scaption"
                processor.rep_count = 0
                processor.state = "NEUTRAL"
                recorded_frames = []
                is_recording = False
                print("[System] Switched to SCAPTION")
            elif key == ord('2'):
                current_exercise = "chin_tuck"
                processor.rep_count = 0
                processor.state = "NEUTRAL"
                recorded_frames = []
                is_recording = False
                print("[System] Switched to CHIN TUCK")

    except KeyboardInterrupt:
        print("Stopping...")    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        socket.close()
        context.term()

if __name__ == "__main__":
    run_server()