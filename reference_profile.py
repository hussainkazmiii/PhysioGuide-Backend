import json
import os


class ReferenceProfile:
    def __init__(self, data):
        self.data = data or {}

        # Metadata
        self.exercise = self.data.get("exercise", "")
        self.video_source = self.data.get(
            "video_source", "none (using hardcoded defaults)"
        )
        self.total_reps = self.data.get("total_reps", 0)
        self.tolerance_multiplier = self.data.get("tolerance_multiplier", 1.0)

        # Core sections
        self.metrics = self.data.get("metrics", {})
        self.errors = self.data.get("errors", {})
        self.state_transitions = self.data.get("state_transitions", {})

    # =========================
    # METRICS
    # =========================
    def get_reference_value(self, metric_name, state):
        """
        Get reference metric value for a given state.
        Example: SEA in HOLDING state
        """
        metric = self.metrics.get(metric_name, {})
        states = metric.get("states", {})
        return states.get(state)

    # =========================
    # ERRORS
    # =========================
    def get_error_max(self, error_name):
        error = self.errors.get(error_name, {})
        return error.get("max")

    def get_error_min(self, error_name):
        error = self.errors.get(error_name, {})
        return error.get("min")

    # =========================
    # STATE TRANSITIONS
    # =========================
    def get_state_transition(self, name):
        return self.state_transitions.get(name)


# ==========================================
# LOAD PROFILE FROM JSON
# ==========================================
def load_profile(path):
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return ReferenceProfile(data)
    except Exception as e:
        print(f"[Error] Failed to load profile: {e}")
        return None


# ==========================================
# DEFAULT PROFILES (FALLBACK)
# ==========================================
def get_default_profile(exercise):
    """
    Returns hardcoded default profiles if no trained JSON exists
    """

    if exercise == "shoulder_abduction":
        return ReferenceProfile({
            "exercise": "shoulder_abduction",
            "video_source": "none (using hardcoded defaults)",
            "total_reps": 0,
            "tolerance_multiplier": 1.0,

            "metrics": {
                "SEA": {
                    "states": {
                        "NEUTRAL": 20,
                        "LIFTING": 60,
                        "HOLDING": 90,
                        "LOWERING": 60
                    }
                },
                "EA": {
                    "states": {
                        "NEUTRAL": 160,
                        "LIFTING": 160,
                        "HOLDING": 160,
                        "LOWERING": 160
                    }
                }
            },

            "errors": {
                "hiking": {"max": 0.05},
                "bent_elbow": {"min": 140}
            },

            "state_transitions": {
                "NEUTRAL_to_LIFTING": {"SEA_threshold": 30},
                "LIFTING_to_HOLDING": {"SEA_threshold": 85},
                "LOWERING_to_NEUTRAL": {"SEA_threshold": 30},
                "LIFTING_fallback": {"SEA_threshold": 25},
                "HOLDING_fallback": {"SEA_threshold": 70},
                "HOLDING_duration": 0.5
            }
        })

    if exercise == "scaption":
        return get_default_profile("shoulder_abduction")

    elif exercise == "chin_tuck":
        return ReferenceProfile({
            "exercise": "chin_tuck",
            "video_source": "none (using hardcoded defaults)",
            "total_reps": 0,
            "tolerance_multiplier": 1.0,

            "metrics": {
                "HRM": {
                    "states": {
                        "NEUTRAL": 0.5,
                        "RETRACTING": 1.5,
                        "HOLDING": 2.5,
                        "RETURNING": 1.0
                    }
                },
                "NAM": {
                    "states": {
                        "NEUTRAL": 0.0,
                        "RETRACTING": 0.0,
                        "HOLDING": 0.0,
                        "RETURNING": 0.0
                    }
                }
            },

            "errors": {
                "hiking": {"max": 0.03},
                "flexion": {"max": 0.05},
                "extension": {"min": -0.05}
            },

            "state_transitions": {
                "NEUTRAL_to_RETRACTING": {"HRM_threshold": 1.0},
                "RETRACTING_to_HOLDING": {"HRM_threshold": 2.0},
                "RETURNING_to_NEUTRAL": {"HRM_threshold": 0.5},
                "RETRACTING_fallback": {"HRM_threshold": 0.5},
                "HOLDING_fallback": {"HRM_threshold": 1.5},
                "HOLDING_duration": 1.0
            }
        })

    return ReferenceProfile({})
