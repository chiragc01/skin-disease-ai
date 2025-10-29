import pandas as pd
import os

# Path to original CSV
ORIGINAL_CSV = r"C:\Users\Asus\Desktop\skin_disease_ai\data\ham10000\HAM10000_metadata.csv"
PROCESSED_CSV = r"C:\Users\Asus\Desktop\skin_disease_ai\data\ham10000\labels_processed.csv"



# Load original CSV
df = pd.read_csv(ORIGINAL_CSV)

# Keep only required columns
df_processed = df[['image_id', 'dx']]

# Save processed CSV
df_processed.to_csv(PROCESSED_CSV, index=False)
print(f"Processed CSV saved at: {PROCESSED_CSV}")
