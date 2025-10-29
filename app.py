from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('skin_disease_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))
        img = img_to_array(image) / 255.0
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)
        return jsonify({"prediction": int(np.argmax(preds))})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
