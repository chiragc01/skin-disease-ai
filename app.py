from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "saved_models/skin_disease_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print("⚠️ Warning: Model file not found at", MODEL_PATH)

# List of class names (update these as per your model)
CLASSES = ["Eczema", "Psoriasis", "Acne", "Healthy Skin"]

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return jsonify({"message": "Skin Disease Prediction API is Live ✅"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    os.makedirs("uploads", exist_ok=True)
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    try:
        # Preprocess image
        img = load_img(img_path, target_size=(128, 128))  # adjust as per your model
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        pred_class = CLASSES[np.argmax(preds)]

        # Clean up the uploaded image
        os.remove(img_path)

        return jsonify({"predicted_disease": pred_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Entry point
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # ✅ Use Render’s PORT
    app.run(host='0.0.0.0', port=port)
