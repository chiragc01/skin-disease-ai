import os
import cv2
import pandas as pd
from tqdm import tqdm  # optional, shows progress bar

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = "../data/ham10000/"
IMG_DIR = os.path.join(DATA_DIR, "images")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "labels_processed.csv")

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = 128  # resize images to 128x128

# -------------------------------
# Create processed folder if not exists
# -------------------------------
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

# -------------------------------
# Load CSV labels
# -------------------------------
labels_df = pd.read_csv(LABELS_CSV)

# Map disease names to numeric labels
labels_df['dx'] = labels_df['dx'].astype('category')
labels_df['label'] = labels_df['dx'].cat.codes

# Save mapping for reference
label_mapping = dict(enumerate(labels_df['dx'].cat.categories))
print("Class mapping (dx -> label):")
for k, v in label_mapping.items():
    print(f"{v} -> {k}")

# -------------------------------
# Process Images
# -------------------------------
for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    img_name = row['image_id'] + ".jpg"
    img_path = os.path.join(IMG_DIR, img_name)
    save_path = os.path.join(PROCESSED_DIR, img_name)

    if os.path.exists(img_path):
        # Load image
        img = cv2.imread(img_path)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Normalize to 0-1
        img = img / 255.0
        # Convert back to 0-255 for saving as image
        img_to_save = (img * 255).astype('uint8')
        cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
    else:
        print(f"Image not found: {img_name}")

# -------------------------------
# Save updated labels CSV
# -------------------------------
labels_df[['image_id', 'label']].to_csv(OUTPUT_CSV, index=False)
print("Preprocessing complete! Processed images saved in:", PROCESSED_DIR)
print("Updated labels saved as:", OUTPUT_CSV)
