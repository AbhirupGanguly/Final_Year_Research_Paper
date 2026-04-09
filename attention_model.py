import numpy as np
import joblib
import pandas as pd
from collections import deque

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
model = joblib.load("attention_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# SETTINGS
# -----------------------------
THRESHOLD = 0.5
history = deque(maxlen=5)

# -----------------------------
# RULE-BASED SCORE
# -----------------------------
def compute_attention_score(data):
    pose_forward = data.get("pose_forward", 0)
    phone = data.get("phone", 0)
    pose_x = data.get("pose_x", 0)
    pose_y = data.get("pose_y", 0)

    head_stability = 1 / (1 + abs(pose_x) + abs(pose_y))
    phone_factor = 1 - phone

    return (
        0.5 * pose_forward +
        0.3 * phone_factor +
        0.2 * head_stability
    )

# -----------------------------
# OPTIONAL: DISTANCE
# -----------------------------
def calculate_distance(student, teacher):
    return ((student[0] - teacher[0])**2 + (student[1] - teacher[1])**2) ** 0.5

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def predict_attention(data):

    # Optional teacher distance
    if "student_pos" in data and "teacher_pos" in data:
        data["distance"] = calculate_distance(
            data["student_pos"], data["teacher_pos"]
        )

    # Rule-based score
    rule_score = compute_attention_score(data)

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # 🔥 IMPORTANT: Match training features automatically
    model_features = scaler.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    # Scale
    features_scaled = scaler.transform(df)

    # Model prediction
    prob = model.predict_proba(features_scaled)[0][1]

    # Temporal smoothing
    history.append(prob)
    smooth_score = np.mean(history)

    # Confidence
    confidence = "High" if smooth_score > 0.7 or smooth_score < 0.3 else "Medium"

    return {
        "attention_score": float(smooth_score),
        "prediction": "Attentive" if smooth_score < THRESHOLD else "Inattentive",
        "confidence": confidence,
        "rule_score": float(rule_score),
        "raw_model_score": float(prob)
    }