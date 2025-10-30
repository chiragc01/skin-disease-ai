from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "skin_disease_model.h5")
INFO_PATH = os.path.join(BASE_DIR, "data", "disease_info.json")

model = load_model(MODEL_PATH)
with open(INFO_PATH, "r") as f:
    disease_info = json.load(f)

label_mapping = {
    0: 'bkl', 1: 'df', 2: 'mel',
    3: 'nv', 4: 'vasc', 5: 'bcc', 6: 'akiec'
}

@app.route('/')
def home():
    return jsonify({"message": "✅ Skin Disease Prediction API is Live!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = load_img(file, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)
        pred_class = int(np.argmax(pred, axis=1)[0])
        pred_label = label_mapping.get(pred_class, "unknown")
        confidence = float(pred[0][pred_class] * 100)

        info = disease_info.get(pred_label, {})
        return jsonify({
            'predicted_label': pred_label,
            'full_name': info.get("full_name", "Unknown"),
            'confidence': confidence,
            'cause': info.get("cause", "Not specified"),
            'habit': info.get("habit", "Not specified")
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()
