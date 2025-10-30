from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# -------------------------------
# Load model once at startup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "skin_disease_model.h5")

print("⏳ Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Define your class names (update with your real ones)
CLASSES = ["bkl", "df", "mel", "nv", "vasc", "bcc", "akiec"]

@app.route('/')
def home():
    return jsonify({"message": "✅ Skin Disease Prediction API is Live!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    os.makedirs("uploads", exist_ok=True)
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    # Preprocess image
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = CLASSES[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)

    return jsonify({
        "predicted_disease": pred_class,
        "confidence": round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
