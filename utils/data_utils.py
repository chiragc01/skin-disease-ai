# utils/data_utils.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def build_filepath_label_lists(images_dir, labels_csv=None, img_exts=('.jpg','.jpeg','.png')):
    """
    If you have labels.csv that maps filename->label, prefer that.
    Otherwise assumes class subfolders under images_dir.
    """
    filepaths = []
    labels = []
    if labels_csv and os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        # adjust depending on your CSV format (assumed columns 'image_id','label')
        for _, row in df.iterrows():
            fname = row['image_id']
            label = row['label']
            p = os.path.join(images_dir, fname)
            if os.path.exists(p):
                filepaths.append(p)
                labels.append(label)
    else:
        # walk subfolders as classes
        for cls in sorted(os.listdir(images_dir)):
            cls_path = os.path.join(images_dir, cls)
            if os.path.isdir(cls_path):
                for f in os.listdir(cls_path):
                    if f.lower().endswith(img_exts):
                        filepaths.append(os.path.join(cls_path,f))
                        labels.append(cls)
    return np.array(filepaths), np.array(labels)

def train_val_split(filepaths, labels, val_size=0.2, stratify=True, random_state=42):
    strat = labels if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(filepaths, labels, test_size=val_size, random_state=random_state, stratify=strat)
    return X_train, X_val, y_train, y_val

def compute_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))
