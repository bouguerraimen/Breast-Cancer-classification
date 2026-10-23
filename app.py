from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")  # ton modèle (ou Pipeline avec scaler) sauvegardé


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or len(features) != 30:
        return jsonify({"error": "Il faut exactement 30 valeurs numériques."}), 400

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "Valeurs invalides."}), 400

    pred = int(model.predict(X)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(X)[0]))

    # Dans le dataset sklearn : 0 = malignant, 1 = benign
    label = "malignant" if pred == 0 else "benign"
    return jsonify({"prediction": pred, "label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)