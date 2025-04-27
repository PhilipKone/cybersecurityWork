import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Step 1: Load and Inspect Dataset ---
# Load the main CSV file
df = pd.read_csv(r'C:/Users/DELL/OneDrive - Data Intelligence & Swarm Analytics Laboratory/Documents/PHconsult/cybersecurityWork/Dataset(D1)/ID__URLs_labell.csv')
print('Loaded dataset shape:', df.shape)
print(df.head())

# --- Step 2: Data Cleaning ---
# Remove duplicates and rows with missing values in key columns
df = df.drop_duplicates(subset=['id', 'url'])
df = df.dropna(subset=['id', 'url', 'typ'])
print('After cleaning:', df.shape)

# --- Step 3: Dataset Split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['typ'])
print('Train set:', train_df.shape, 'Test set:', test_df.shape)

# --- Step 4: Sample HTML/Raw Content Loading ---
# Example: Load the raw content for the first 5 samples
raw_dir = r'C:/Users/DELL/OneDrive - Data Intelligence & Swarm Analytics Laboratory/Documents/PHconsult/cybersecurityWork/Dataset(D1)/sourc_Data(D1)'

def load_raw_content(row):
    file_path = os.path.join(raw_dir, f"{int(row['id'])}.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return ''
    return ''

sample_train = train_df.head(5).copy()
sample_train['raw_content'] = sample_train.apply(load_raw_content, axis=1)
print('Sample loaded raw content:')
print(sample_train[['id', 'url', 'raw_content']])

# --- Next Steps Guidance ---
# 1. Feature extraction from URL and raw_content (to be implemented next)
# 2. Vectorization and model training
# 3. Evaluation
