import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = r"C:\Users\Asus\Desktop\skin_disease_ai\data\ham10000"
IMG_DIR = os.path.join(DATA_DIR, "images")  # images folder
LABELS_CSV = os.path.join(DATA_DIR, "labels_processed.csv")
MODEL_SAVE_PATH = r"C:\Users\Asus\Desktop\skin_disease_ai\saved_models\skin_disease_model.h5"

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = 128  # must match preprocess.py
BATCH_SIZE = 32
EPOCHS = 20

# -------------------------------
# Load labels
# -------------------------------
labels_df = pd.read_csv(LABELS_CSV)
NUM_CLASSES = len(labels_df['dx'].unique())
print(f"Number of classes: {NUM_CLASSES}")

# Create mapping from disease name -> integer
label_mapping = {label: idx for idx, label in enumerate(labels_df['dx'].unique())}
print("Label mapping:", label_mapping)

# -------------------------------
# Load images and labels
# -------------------------------
X = []
y = []

for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    img_name = row['image_id'] + ".jpg"  # change to .png if your images are PNG
    img_path = os.path.join(IMG_DIR, img_name)
    
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0  # normalize
        X.append(img_array)
        y.append(label_mapping[row['dx']])
    else:
        print(f"Image not found: {img_name}")

X = np.array(X, dtype='float32')
y = to_categorical(y, NUM_CLASSES)  # one-hot encode

print("Images shape:", X.shape)
print("Labels shape:", y.shape)

# -------------------------------
# Train-validation split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# -------------------------------
# Build CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -------------------------------
# Save Model
# -------------------------------
save_dir = os.path.dirname(MODEL_SAVE_PATH)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model.save(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")
    