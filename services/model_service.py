from pathlib import Path
import json
import joblib
import tensorflow as tf


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ---------------------------------------------------------
# LOAD RANDOM FOREST
# ---------------------------------------------------------


RF_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"

rf_model = joblib.load(RF_MODEL_PATH)



# ---------------------------------------------------------
# LOAD LSTM
# ---------------------------------------------------------

LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"

lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)



# ---------------------------------------------------------
# LOAD LSTM PREPROCESSING
# ---------------------------------------------------------

LSTM_SCALER_PATH = MODEL_DIR / "lstm_scaler.joblib"
LSTM_ENCODER_PATH = MODEL_DIR / "lstm_label_encoder.joblib"

lstm_scaler = joblib.load(LSTM_SCALER_PATH)
lstm_label_encoder = joblib.load(LSTM_ENCODER_PATH)



# ---------------------------------------------------------
# LOAD RF FEATURE NAMES
# ---------------------------------------------------------

RF_FEATURE_NAMES_PATH = MODEL_DIR / "rf_feature_names.json"

with open(RF_FEATURE_NAMES_PATH, "r") as f:
    rf_feature_names = json.load(f)


# ---------------------------------------------------------
# LOAD RF TRAINING MEDIANS
# ---------------------------------------------------------

RF_MEDIANS_PATH = MODEL_DIR / "rf_training_medians.json"

with open(RF_MEDIANS_PATH, "r") as f:
    rf_training_medians = json.load(f)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

RF_WEIGHT = 0.6
LSTM_WEIGHT = 0.4

LSTM_SEQUENCE_LENGTH = 30

CLASSES = ["BUY", "HOLD", "SELL"]

TARGET_HORIZON_DAYS = 5
TARGET_THRESHOLD_PERCENT = 1.5


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

def get_model_info():
    return {
        "random_forest": {
            "loaded": rf_model is not None,
            "features": len(rf_feature_names),
            "trees": rf_model.n_estimators,
            "weight": RF_WEIGHT,
        },
        "lstm": {
            "loaded": lstm_model is not None,
            "sequence_length": LSTM_SEQUENCE_LENGTH,
            "features": lstm_scaler.n_features_in_,
            "weight": LSTM_WEIGHT,
            "classes": list(lstm_label_encoder.classes_),
        },
        "classes": CLASSES,
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "target_threshold_percent": TARGET_THRESHOLD_PERCENT,
    }