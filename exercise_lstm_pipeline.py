from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

TIMESTEPS = 10
LANDMARK_NAMES = [
    "sh_L",
    "sh_R",
    "el_L",
    "el_R",
    "wr_L",
    "wr_R",
    "hi_L",
    "hi_R",
    "kn_L",
    "kn_R",
]
RAW_FEATURES_PER_FRAME = len(LANDMARK_NAMES) * 3
RAW_FEATURES_TOTAL = TIMESTEPS * RAW_FEATURES_PER_FRAME

BIOMECH_FEATURE_NAMES = [
    "left_shoulder_elevation",
    "right_shoulder_elevation",
    "left_elbow_angle",
    "right_elbow_angle",
    "shoulder_symmetry",
    "shoulder_y_delta",
    "left_wrist_height",
    "right_wrist_height",
]
FEATURES_PER_FRAME = (RAW_FEATURES_PER_FRAME * 2) + len(BIOMECH_FEATURE_NAMES)

LABELS = (
    "chin_tuck_correct",
    "chin_tuck_incorrect",
    "shoulder_abduction_correct",
    "shoulder_abduction_incorrect",
)
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

EXERCISE_ALIASES = {
    "chin tuck": "chin_tuck",
    "chin_tuck": "chin_tuck",
    "scaption": "shoulder_abduction",
    "shoulder abduction": "shoulder_abduction",
    "shoulder_abduction": "shoulder_abduction",
}

POSE_LANDMARK_INDICES = {
    "sh_L": 11,
    "sh_R": 12,
    "el_L": 13,
    "el_R": 14,
    "wr_L": 15,
    "wr_R": 16,
    "hi_L": 23,
    "hi_R": 24,
    "kn_L": 25,
    "kn_R": 26,
}

MODEL_PATH = BASE_DIR / "exercise_lstm_model.keras"
SCALER_PATH = BASE_DIR / "exercise_scaler.json"
METADATA_PATH = BASE_DIR / "exercise_label_map.json"


def expected_feature_columns() -> List[str]:
    columns: List[str] = []
    for frame_idx in range(TIMESTEPS):
        for landmark in LANDMARK_NAMES:
            for axis in ("x", "y", "z"):
                columns.append(f"f{frame_idx}_{landmark}_{axis}")
    return columns


FEATURE_COLUMNS = expected_feature_columns()


def normalize_exercise_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    key = "_".join(key.split())
    return EXERCISE_ALIASES.get(key, key)


def normalize_label_name(label: str) -> str:
    return str(label).strip().lower().replace("-", "_")


def class_name_from_parts(exercise: str, label: str) -> str:
    return f"{normalize_exercise_name(exercise)}_{normalize_label_name(label)}"


def split_class_name(class_name: str) -> Tuple[str, str]:
    if class_name.endswith("_correct"):
        return class_name[: -len("_correct")], "correct"
    if class_name.endswith("_incorrect"):
        return class_name[: -len("_incorrect")], "incorrect"
    return class_name, "unknown"


def display_exercise_name(exercise: str) -> str:
    exercise = normalize_exercise_name(exercise)
    return exercise.replace("_", " ").title()


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Missing {len(missing)} feature columns. First missing: {preview}")

    extra = [
        column
        for column in df.columns
        if column not in FEATURE_COLUMNS and column not in {"label", "exercise"}
    ]
    if extra:
        preview = ", ".join(extra[:8])
        raise ValueError(f"Unexpected feature columns found. First unexpected: {preview}")


def load_labeled_csvs(csv_paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        validate_feature_columns(df)
        if "label" not in df.columns or "exercise" not in df.columns:
            raise ValueError(f"{csv_path} must contain label and exercise columns")

        df = df.copy()
        df["exercise"] = df["exercise"].map(normalize_exercise_name)
        df["label"] = df["label"].map(normalize_label_name)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    class_names = data.apply(
        lambda row: class_name_from_parts(row["exercise"], row["label"]),
        axis=1,
    )
    unknown = sorted(set(class_names) - set(LABEL_TO_INDEX))
    if unknown:
        raise ValueError(f"Unknown class labels in data: {unknown}")

    X_flat = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if np.isnan(X_flat).any():
        nan_count = int(np.isnan(X_flat).sum())
        raise ValueError(f"Feature matrix contains {nan_count} NaN values")

    y = class_names.map(LABEL_TO_INDEX).to_numpy(np.int64)
    return X_flat, y, data


def reshape_flat_features(X_flat: np.ndarray) -> np.ndarray:
    X_flat = np.asarray(X_flat, dtype=np.float32)
    if X_flat.ndim != 2 or X_flat.shape[1] != RAW_FEATURES_TOTAL:
        raise ValueError(
            f"Expected flat features shaped (n, {RAW_FEATURES_TOTAL}), got {X_flat.shape}"
        )
    return X_flat.reshape(X_flat.shape[0], TIMESTEPS, RAW_FEATURES_PER_FRAME)


def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    denom = np.maximum(denom, 1e-8)
    cosine = np.sum(ba * bc, axis=-1) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _landmark(points: np.ndarray, name: str) -> np.ndarray:
    return points[:, :, LANDMARK_NAMES.index(name), :]


def normalize_landmark_sequences(X_seq: np.ndarray) -> np.ndarray:
    points = np.asarray(X_seq, dtype=np.float32).reshape(
        X_seq.shape[0], TIMESTEPS, len(LANDMARK_NAMES), 3
    )

    sh_l = _landmark(points, "sh_L")
    sh_r = _landmark(points, "sh_R")
    hi_l = _landmark(points, "hi_L")
    hi_r = _landmark(points, "hi_R")

    mid_hip = (hi_l + hi_r) / 2.0
    mid_shoulder = (sh_l + sh_r) / 2.0
    shoulder_width = np.linalg.norm(sh_l - sh_r, axis=-1)
    torso_length = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)
    scale = np.maximum.reduce([shoulder_width, torso_length, np.full_like(shoulder_width, 1e-4)])

    normalized = (points - mid_hip[:, :, None, :]) / scale[:, :, None, None]
    return normalized.astype(np.float32)


def biomech_features(normalized_points: np.ndarray) -> np.ndarray:
    sh_l = _landmark(normalized_points, "sh_L")
    sh_r = _landmark(normalized_points, "sh_R")
    el_l = _landmark(normalized_points, "el_L")
    el_r = _landmark(normalized_points, "el_R")
    wr_l = _landmark(normalized_points, "wr_L")
    wr_r = _landmark(normalized_points, "wr_R")
    hi_l = _landmark(normalized_points, "hi_L")
    hi_r = _landmark(normalized_points, "hi_R")

    left_shoulder = angle_3d(hi_l, sh_l, el_l)
    right_shoulder = angle_3d(hi_r, sh_r, el_r)
    left_elbow = angle_3d(sh_l, el_l, wr_l)
    right_elbow = angle_3d(sh_r, el_r, wr_r)
    shoulder_symmetry = np.abs(left_shoulder - right_shoulder)
    shoulder_y_delta = np.abs(sh_l[..., 1] - sh_r[..., 1])

    # MediaPipe image-space y increases downward, so shoulder_y - wrist_y is
    # positive when the wrist rises toward or above shoulder height.
    left_wrist_height = sh_l[..., 1] - wr_l[..., 1]
    right_wrist_height = sh_r[..., 1] - wr_r[..., 1]

    return np.stack(
        [
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
            shoulder_symmetry,
            shoulder_y_delta,
            left_wrist_height,
            right_wrist_height,
        ],
        axis=-1,
    ).astype(np.float32)


def build_model_features(X_seq: np.ndarray) -> np.ndarray:
    X_seq = np.asarray(X_seq, dtype=np.float32)
    if X_seq.ndim == 2:
        X_seq = X_seq.reshape(1, TIMESTEPS, RAW_FEATURES_PER_FRAME)
    if X_seq.shape[1:] != (TIMESTEPS, RAW_FEATURES_PER_FRAME):
        raise ValueError(
            f"Expected sequence shape (n, {TIMESTEPS}, {RAW_FEATURES_PER_FRAME}), got {X_seq.shape}"
        )

    normalized_points = normalize_landmark_sequences(X_seq)
    normalized_flat = normalized_points.reshape(X_seq.shape[0], TIMESTEPS, RAW_FEATURES_PER_FRAME)

    velocity = np.diff(normalized_flat, axis=1, prepend=normalized_flat[:, :1, :])
    biomech = biomech_features(normalized_points)

    features = np.concatenate([normalized_flat, velocity, biomech], axis=-1)
    if features.shape[-1] != FEATURES_PER_FRAME:
        raise RuntimeError(f"Unexpected feature size {features.shape[-1]}")
    return features.astype(np.float32)


@dataclass
class SequenceStandardScaler:
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SequenceStandardScaler":
        flat = np.asarray(X, dtype=np.float32).reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.scale_ = flat.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted")
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted")
        payload = {
            "mean": self.mean_.astype(float).tolist(),
            "scale": self.scale_.astype(float).tolist(),
            "features_per_frame": FEATURES_PER_FRAME,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SequenceStandardScaler":
        payload = json.loads(path.read_text(encoding="utf-8"))
        scaler = cls(
            mean_=np.asarray(payload["mean"], dtype=np.float32),
            scale_=np.asarray(payload["scale"], dtype=np.float32),
        )
        if scaler.mean_.shape[0] != FEATURES_PER_FRAME:
            raise ValueError(
                f"Scaler expects {scaler.mean_.shape[0]} features, code expects {FEATURES_PER_FRAME}"
            )
        return scaler


def make_grouped_split(
    y: np.ndarray,
    class_names: Sequence[str],
    group_size: int,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    group_size = max(1, int(group_size))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for class_idx, class_name in enumerate(class_names):
        indices = np.flatnonzero(y == class_idx)
        if len(indices) == 0:
            raise ValueError(f"No samples found for class {class_name}")

        groups = [indices[start : start + group_size] for start in range(0, len(indices), group_size)]
        rng.shuffle(groups)

        n_groups = len(groups)
        n_test = max(1, int(round(n_groups * test_size)))
        n_val = max(1, int(round(n_groups * val_size)))
        if n_test + n_val >= n_groups:
            n_test = max(1, min(n_test, n_groups - 2))
            n_val = max(1, min(n_val, n_groups - n_test - 1))

        test_groups = groups[:n_test]
        val_groups = groups[n_test : n_test + n_val]
        train_groups = groups[n_test + n_val :]

        train_idx.extend(np.concatenate(train_groups).tolist())
        val_idx.extend(np.concatenate(val_groups).tolist())
        test_idx.extend(np.concatenate(test_groups).tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def class_distribution(y: np.ndarray) -> Dict[str, int]:
    return {
        INDEX_TO_LABEL[idx]: int(np.sum(np.asarray(y) == idx))
        for idx in range(len(LABELS))
    }


def class_weights(y: np.ndarray) -> Dict[int, float]:
    y = np.asarray(y)
    total = len(y)
    weights = {}
    for idx in range(len(LABELS)):
        count = int(np.sum(y == idx))
        weights[idx] = float(total / (len(LABELS) * max(count, 1)))
    return weights


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        cm[int(true_idx), int(pred_idx)] += 1
    return cm


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    lines = ["class,precision,recall,f1,support"]
    for idx, label in INDEX_TO_LABEL.items():
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        support = int(np.sum(y_true == idx))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        lines.append(f"{label},{precision:.4f},{recall:.4f},{f1:.4f},{support}")
    return "\n".join(lines)


def metadata_payload() -> Dict[str, object]:
    return {
        "label_to_index": LABEL_TO_INDEX,
        "index_to_label": {str(idx): label for idx, label in INDEX_TO_LABEL.items()},
        "timesteps": TIMESTEPS,
        "landmark_names": LANDMARK_NAMES,
        "raw_features_per_frame": RAW_FEATURES_PER_FRAME,
        "biomech_feature_names": BIOMECH_FEATURE_NAMES,
        "features_per_frame": FEATURES_PER_FRAME,
        "feature_layout": "normalized_landmarks + temporal_velocity + biomech_features",
        "model_input_shape": [TIMESTEPS, FEATURES_PER_FRAME],
        "exercise_aliases": EXERCISE_ALIASES,
    }


def extract_frame_from_pose_landmarks(landmarks: Sequence[object]) -> Optional[np.ndarray]:
    if landmarks is None:
        return None

    values: List[float] = []
    try:
        for name in LANDMARK_NAMES:
            lm = landmarks[POSE_LANDMARK_INDICES[name]]
            values.extend([float(lm.x), float(lm.y), float(lm.z)])
    except (IndexError, AttributeError, TypeError, ValueError):
        return None
    return np.asarray(values, dtype=np.float32)


def prediction_message(
    expected_exercise: str,
    predicted_exercise: str,
    correctness: str,
    confidence: float,
) -> str:
    expected = normalize_exercise_name(expected_exercise)
    predicted = normalize_exercise_name(predicted_exercise)
    if confidence < 0.60:
        return "Movement unclear. Keep your full upper body visible and repeat slowly."
    if predicted != expected:
        return (
            f"Wrong exercise detected. The model sees {display_exercise_name(predicted)}; "
            f"perform {display_exercise_name(expected)}."
        )
    if correctness == "correct":
        return "Correct rep pattern detected."
    return "Incorrect rep pattern detected. Follow the form corrections."


PERFECT_REP_TIPS = {
    "chin_tuck": [
        "Keep shoulders relaxed and down.",
        "Look straight ahead and glide the head backward.",
        "Avoid nodding the chin down or tilting the head up.",
        "Move slowly, hold briefly, then return with control.",
    ],
    "shoulder_abduction": [
        "Keep elbows mostly straight while raising the arms.",
        "Lift smoothly out to the side until near shoulder height.",
        "Do not shrug; keep shoulders relaxed and level.",
        "Lower with control instead of dropping the arms.",
    ],
}


def perfect_rep_tips(exercise: str) -> List[str]:
    return PERFECT_REP_TIPS.get(normalize_exercise_name(exercise), [])


class ExerciseLSTMClassifier:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        metadata_path: Path = METADATA_PATH,
        confidence_threshold: float = 0.60,
        correct_threshold: float = 0.70,
        incorrect_threshold: float = 0.30,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)
        self.confidence_threshold = confidence_threshold
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold
        self.buffers: Dict[str, deque[np.ndarray]] = {}
        self.model = None
        self.scaler: Optional[SequenceStandardScaler] = None
        self.enabled = False
        self.error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists() or not self.scaler_path.exists():
            self.error = "Model artifacts not found"
            return

        try:
            import tensorflow as tf  # Imported lazily so the server can run without artifacts.

            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = SequenceStandardScaler.load(self.scaler_path)
            self.enabled = True
        except Exception as exc:
            self.error = str(exc)
            self.enabled = False

    def reset(self, exercise: Optional[str] = None) -> None:
        if exercise is None:
            self.buffers.clear()
            return
        self.buffers.pop(normalize_exercise_name(exercise), None)

    def predict_from_frame(
        self,
        exercise: str,
        landmarks: Sequence[object],
    ) -> Dict[str, object]:
        exercise = normalize_exercise_name(exercise)
        if not self.enabled or self.model is None or self.scaler is None:
            return {
                "enabled": False,
                "ready": False,
                "error": self.error,
            }

        frame = extract_frame_from_pose_landmarks(landmarks)
        if frame is None:
            return {
                "enabled": True,
                "ready": False,
                "error": "Could not extract required pose landmarks",
            }

        buffer = self.buffers.setdefault(exercise, deque(maxlen=TIMESTEPS))
        buffer.append(frame)
        if len(buffer) < TIMESTEPS:
            return {
                "enabled": True,
                "ready": False,
                "frames_collected": len(buffer),
                "frames_required": TIMESTEPS,
            }

        X_seq = np.asarray(buffer, dtype=np.float32).reshape(1, TIMESTEPS, RAW_FEATURES_PER_FRAME)
        X_features = build_model_features(X_seq)
        X_scaled = self.scaler.transform(X_features)
        probs = self.model.predict(X_scaled, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        class_name = INDEX_TO_LABEL[pred_idx]
        predicted_exercise, correctness = split_class_name(class_name)
        confidence = float(probs[pred_idx])

        expected_correct_key = f"{exercise}_correct"
        expected_incorrect_key = f"{exercise}_incorrect"
        expected_correct_prob = float(probs[LABEL_TO_INDEX[expected_correct_key]])
        expected_incorrect_prob = float(probs[LABEL_TO_INDEX[expected_incorrect_key]])
        wrong_exercise = predicted_exercise != exercise and confidence >= self.confidence_threshold

        if wrong_exercise:
            decision_correctness = "wrong_exercise"
            is_correct = False
        elif expected_incorrect_prob >= self.incorrect_threshold:
            decision_correctness = "incorrect"
            is_correct = False
        elif expected_correct_prob >= self.correct_threshold:
            decision_correctness = "correct"
            is_correct = True
        else:
            decision_correctness = "uncertain"
            is_correct = False

        return {
            "enabled": True,
            "ready": True,
            "expected_exercise": exercise,
            "predicted_exercise": predicted_exercise,
            "correctness": correctness,
            "decision_correctness": decision_correctness,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "expected_correct_probability": round(expected_correct_prob, 4),
            "expected_incorrect_probability": round(expected_incorrect_prob, 4),
            "is_correct": bool(is_correct),
            "message": prediction_message(
                exercise,
                predicted_exercise,
                "correct" if is_correct else "incorrect",
                confidence,
            ),
            "probabilities": {
                INDEX_TO_LABEL[idx]: round(float(prob), 4)
                for idx, prob in enumerate(probs)
            },
        }
