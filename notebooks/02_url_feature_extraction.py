import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import re
from collections import Counter
from scipy.stats import entropy

# --- Step 1: Load and Inspect Dataset ---
df = pd.read_csv(r'C:/Users/DELL/OneDrive - Data Intelligence & Swarm Analytics Laboratory/Documents/PHconsult/cybersecurityWork/Dataset(D1)/ID__URLs_labell.csv')

# --- Step 2: Data Cleaning ---
df = df.drop_duplicates(subset=['id', 'url'])
df = df.dropna(subset=['id', 'url', 'typ'])

# --- Step 3: Dataset Split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['typ'])

# --- Step 4: URL Feature Extraction ---
def url_length(url):
    return len(url)

def count_special_chars(url):
    return sum([1 for c in url if not c.isalnum()])

def digit_ratio(url):
    digits = sum(c.isdigit() for c in url)
    return digits / len(url) if len(url) > 0 else 0

def letter_ratio(url):
    letters = sum(c.isalpha() for c in url)
    return letters / len(url) if len(url) > 0 else 0

def count_subdomains(url):
    try:
        netloc = urlparse('http://' + url).netloc
        return netloc.count('.')
    except:
        return 0

def url_entropy(url):
    prob = [n_x/len(url) for x,n_x in Counter(url).items()]
    return entropy(prob)

def has_ip_address(url):
    # Returns 1 if the URL contains an IP address, else 0
    return int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url)))

def extract_url_features(df):
    features = pd.DataFrame()
    features['url_length'] = df['url'].apply(url_length)
    features['special_char_count'] = df['url'].apply(count_special_chars)
    features['digit_ratio'] = df['url'].apply(digit_ratio)
    features['letter_ratio'] = df['url'].apply(letter_ratio)
    features['subdomain_count'] = df['url'].apply(count_subdomains)
    features['url_entropy'] = df['url'].apply(url_entropy)
    features['has_ip'] = df['url'].apply(has_ip_address)
    return features

# Extract URL features for training and test sets
X_train_url = extract_url_features(train_df)
X_test_url = extract_url_features(test_df)
y_train = train_df['typ'].astype(int)
y_test = test_df['typ'].astype(int)

print('Sample URL features:')
print(X_train_url.head())

# Save features for next steps
X_train_url.to_csv('notebooks/X_train_url_features.csv', index=False)
X_test_url.to_csv('notebooks/X_test_url_features.csv', index=False)
pd.DataFrame({'y_train': y_train}).to_csv('notebooks/y_train.csv', index=False)
pd.DataFrame({'y_test': y_test}).to_csv('notebooks/y_test.csv', index=False)
