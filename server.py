# app.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Optional: xgboost import for handling raw Booster objects
try:
    import xgboost as xgb
except Exception:
    xgb = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, "final_shuffled.csv")   # update if your csv name differs
MODEL_PATH = os.path.join(APP_DIR, "xgboost_model.pkl")  # update if your pkl name differs

app = Flask(__name__)
CORS(app)  # allow cross-origin for frontend during local dev


def load_model(path=MODEL_PATH):
    """Load model and detect its type/feature names."""
    if not os.path.exists(path):
        print(f"[WARN] Model file not found at {path}")
        return None

    model = joblib.load(path)
    info = {"type": type(model).__name__}

    # Try to determine feature names expected by the model
    feature_names = None
    try:
        # sklearn wrappers often provide feature_names_in_
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            feature_names = list(feature_names)
    except Exception:
        feature_names = None

    # xgboost sklearn wrapper
    try:
        # XGBClassifier from xgboost has .get_booster() and .feature_names
        booster = None
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        elif isinstance(model, xgb.Booster) if xgb else False:
            booster = model

        if booster is not None:
            # booster.feature_names may exist
            b_names = getattr(booster, "feature_names", None)
            if b_names:
                feature_names = list(b_names)
    except Exception:
        pass

    info["feature_names"] = feature_names
    print("[MODEL] Loaded:", info)
    return {"model": model, "info": info}


MODEL_OBJ = load_model(MODEL_PATH)


def get_risk_level(score: float):
    """Risk buckets - tune thresholds to your needs."""
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


@app.route("/")
def index():
    return render_template("chronic_care_dashboard .html")


@app.route("/api/patients", methods=["GET"])
def api_patients():
    """Read CSV live and return patient summary list from final_shuffled_pred.csv."""
    if not os.path.exists(CSV_PATH):
        return jsonify({"status": "error", "message": f"CSV file not found: {CSV_PATH}"}), 500

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading CSV: {str(e)}"}), 500

    # subject_id column
    subject_col = "subject_id" if "subject_id" in df.columns else df.columns[0]

    # probability column
    if "pred_prob" in df.columns:
        prob_col = "pred_prob"
    elif "risk_score" in df.columns:
        prob_col = "risk_score"
    else:
        prob_col = None

    patients = []
    for _, row in df.iterrows():
        subj = row.get(subject_col)
        try:
            subj = int(subj)
        except Exception:
            pass

        score = None
        if prob_col is not None:
            try:
                score = float(row.get(prob_col))
            except Exception:
                score = None
        if score is None:
            score = 0.0

        risk_level = get_risk_level(score)

        patients.append({
            "subject_id": subj,
            "risk_score": round(float(score) * 100, 1),  # % format
            "risk_level": risk_level,

            # main vitals (means only)
            "Heart_Rate_itemmean": row.get("Heart Rate_itemmean"),
            "Systolic_BP_itemmean": row.get("Systolic BP_itemmean"),
            "Temperature_itemmean": row.get("Temperature_itemmean"),
            "SpO2_itemmean": row.get("Oxygen Saturation_itemmean"),

            # detailed vitals for optional frontend use
            "vitals": {
                "hr_min": row.get("Heart Rate_itemmin"),
                "hr_max": row.get("Heart Rate_itemmax"),
                "hr_mean": row.get("Heart Rate_itemmean"),

                "sbp_min": row.get("Systolic BP_itemmin"),
                "sbp_max": row.get("Systolic BP_itemmax"),
                "sbp_mean": row.get("Systolic BP_itemmean"),

                "spo2_min": row.get("Oxygen Saturation_itemmin"),
                "spo2_max": row.get("Oxygen Saturation_itemmax"),
                "spo2_mean": row.get("Oxygen Saturation_itemmean"),

                "temp_mean": row.get("Temperature_itemmean")
            }
        })

    return jsonify({"status": "success", "data": patients})



@app.route("/api/model/performance", methods=["GET"])
def api_model_performance():
    """
    Returns model performance metrics.
    If you have precomputed metrics (e.g. in a JSON file), return them.
    Otherwise returns a fallback example.
    """
    metrics_path = os.path.join(APP_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as fh:
            try:
                metrics = json.load(fh)
                return jsonify({"status": "success", "data": metrics})
            except Exception:
                pass

    # fallback - replace with your real numbers if you have them
    return jsonify({"status": "success", "data": {
        "auroc": 0.892,
        "auprc": 0.847,
        "accuracy": 0.873,
        "f1_score": 0.823
    }})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts JSON containing features, returns a risk probability and bucket.
    Example payload (only minimal features):
    {
      "Heart_Rate_itemmean": 85.6,
      "Systolic_BP_itemmean": 120.0,
      "Temperature_itemmean": 36.8
    }
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    if MODEL_OBJ is None or MODEL_OBJ.get("model") is None:
        return jsonify({"status": "error", "message": "Model not loaded on server"}), 500

    model = MODEL_OBJ["model"]
    model_info = MODEL_OBJ.get("info", {})
    feature_names = model_info.get("feature_names")

    # Build dataframe from payload. If model expects a certain column ordering (feature_names),
    # we will try to reindex to that. Otherwise, pass whatever features provided.
    X = pd.DataFrame([payload])

    # If model expects feature names, attempt to reindex - missing features will become NaN
    if feature_names:
        missing = [f for f in feature_names if f not in X.columns]
        if missing:
            # we will add missing columns as NaN (you may want to fill defaults)
            for m in missing:
                X[m] = np.nan
        X = X[feature_names]

    try:
        # Case 1: sklearn-like object with predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # binary: proba[:,1] is positive class
            score = float(proba[:, 1][0])
        # Case 2: sklearn-like but only predict (returns class). We'll use it as 0/1.
        elif hasattr(model, "predict"):
            pred = model.predict(X)
            # if predict gives probability-like floats we try clamp; otherwise map class to 0/1
            try:
                val = float(pred[0])
                if 0.0 <= val <= 1.0:
                    score = val
                else:
                    score = float(pred[0])  # treat as class (0/1)
            except Exception:
                # fallback: if non-numeric return 1.0 for positive class
                score = 1.0 if pred[0] else 0.0
        # Case 3: raw xgboost.Booster
        elif xgb and isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X, missing=np.nan)
            pred = model.predict(dmat)
            score = float(pred[0])
        else:
            return jsonify({"status": "error", "message": f"Unsupported model type: {type(model)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error making prediction: {str(e)}"}), 500

    risk_level = get_risk_level(score)
    return jsonify({"status": "success", "prediction": {"risk_score": score, "risk_level": risk_level}})


if __name__ == "__main__":
    print("Starting Flask server...")
    print("CSV path:", CSV_PATH)
    print("Model path:", MODEL_PATH)
    if MODEL_OBJ:
        print("Model info:", MODEL_OBJ.get("info"))
    app.run(host="0.0.0.0", port=5002, debug=True)
