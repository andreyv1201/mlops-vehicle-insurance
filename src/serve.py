import os
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = None
scaler = None


def load_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"[Этап 7]Модель загружена из {MODEL_PATH}")
    else:
        print("[Этап 7] Модель не найдена, Сначала запустите main.py")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    return jsonify({"premium": round(float(prediction), 2)})


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    features = np.array(data["features"])
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)

    return jsonify({"premiums": [round(float(p), 2) for p in predictions]})


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
