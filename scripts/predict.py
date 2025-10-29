import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json

# -------------------------------
# Auto-detect base directory (where this script is)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "skin_disease_model.h5")
INFO_PATH = os.path.join(BASE_DIR, "..", "data", "disease_info.json")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "..", "data", "test_images")

IMG_SIZE = 128  # must match training script

# -------------------------------
# Debug info
# -------------------------------
print(f"🧭 MODEL_PATH = {os.path.abspath(MODEL_PATH)}")
print(f"🧭 INFO_PATH  = {os.path.abspath(INFO_PATH)}")
print(f"🧭 TEST_IMAGE_DIR = {os.path.abspath(TEST_IMAGE_DIR)}")

# -------------------------------
# Verify file existence
# -------------------------------
if not os.path.exists(INFO_PATH):
    raise FileNotFoundError(f"❌ Disease info file not found: {INFO_PATH}\nPlease verify location!")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}\nPlease verify location!")

if not os.path.exists(TEST_IMAGE_DIR):
    raise FileNotFoundError(f"❌ test_images folder not found: {TEST_IMAGE_DIR}\nPlease verify location!")

# -------------------------------
# Load model
# -------------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# -------------------------------
# Label mapping
# -------------------------------
label_mapping = {
    0: 'bkl',
    1: 'df',
    2: 'mel',
    3: 'nv',
    4: 'vasc',
    5: 'bcc',
    6: 'akiec'
}

# -------------------------------
# Load disease info (JSON)
# -------------------------------
with open(INFO_PATH, "r") as f:
    disease_info = json.load(f)

# -------------------------------
# Load and loop through all test images
# -------------------------------
image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    raise FileNotFoundError(f"❌ No images found in folder: {TEST_IMAGE_DIR}")

print(f"✅ Found {len(image_files)} image(s) in test_images folder.\n")

for img_name in image_files:
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)
    print(f"🔍 Processing: {img_name}")

    # Load and preprocess image
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    pred = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    pred_label = label_mapping.get(pred_class, "unknown")
    confidence = pred[0][pred_class] * 100

    # Fetch full disease info
    info = disease_info.get(pred_label, {})
    name = info.get("full_name", "Unknown Disease")
    cause = info.get("cause", "Cause not specified.")
    habit = info.get("habit", "Habit not specified.")

    # Display results
    print("\n==================== RESULT ====================")
    print(f"🧠 Predicted Disease Code : {pred_label}")
    print(f"🩺 Full Disease Name     : {name}")
    print(f"📊 Confidence            : {confidence:.2f}%")
    print(f"\n⚠️  Cause: {cause}")
    print(f"🚫 Habit/Trigger: {habit}")
    print("\n💡 Advice: Consult a dermatologist for accurate diagnosis.")
    print("================================================\n")

print("🎯 All images processed successfully!")
