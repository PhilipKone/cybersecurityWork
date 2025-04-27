import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Suppress XMLParsedAsHTMLWarning if parsing XML with HTML parser
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- Step 1: Load Data Splits ---
train_df = pd.read_csv('notebooks/y_train.csv')
test_df = pd.read_csv('notebooks/y_test.csv')
train_ids = pd.read_csv(r'C:/Users/DELL/OneDrive - Data Intelligence & Swarm Analytics Laboratory/Documents/PHconsult/cybersecurityWork/Dataset(D1)/ID__URLs_labell.csv').iloc[train_df.index]
test_ids = pd.read_csv(r'C:/Users/DELL/OneDrive - Data Intelligence & Swarm Analytics Laboratory/Documents/PHconsult/cybersecurityWork/Dataset(D1)/ID__URLs_labell.csv').iloc[test_df.index]

raw_dir = r'C:/Users/DELL/OneDrive - Data Intelligence & Swarm Analytics Laboratory/Documents/PHconsult/cybersecurityWork/Dataset(D1)/sourc_Data(D1)'

def load_html_content(row):
    file_path = os.path.join(raw_dir, f"{int(row['id'])}.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return ''
    return ''

def extract_plaintext(html):
    soup = BeautifulSoup(html, 'lxml')
    # Remove scripts and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=' ', strip=True)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_hyperlink_stats(html):
    soup = BeautifulSoup(html, 'lxml')
    links = soup.find_all('a')
    num_links = len(links)
    num_external = sum(1 for a in links if a.get('href') and 'http' in a.get('href'))
    num_internal = num_links - num_external
    num_imgs = len(soup.find_all('img'))
    num_scripts = len(soup.find_all('script'))
    num_forms = len(soup.find_all('form'))
    return pd.Series({
        'num_links': num_links,
        'num_external_links': num_external,
        'num_internal_links': num_internal,
        'num_imgs': num_imgs,
        'num_scripts': num_scripts,
        'num_forms': num_forms
    })

# --- Step 2: Load and Process HTML Content ---
def load_html_content_by_id(id_):
    file_path = os.path.join(raw_dir, f"{int(id_)}.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ''
    return ''

def parallel_load_html(ids, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        raw_htmls = list(tqdm(executor.map(load_html_content_by_id, ids), total=len(ids), desc='Loading HTML'))
    return raw_htmls

import traceback

if __name__ == "__main__":
    try:
        train_html = train_ids.copy()
        test_html = test_ids.copy()
        train_html['raw_html'] = parallel_load_html(train_html['id'])
        test_html['raw_html'] = parallel_load_html(test_html['id'])

        # Helper for parallel feature extraction
        from concurrent.futures import ProcessPoolExecutor

        def parallel_apply(func, data, max_workers=8, desc=None):
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(func, data), total=len(data), desc=desc))
            return results

        # Extract plaintext for TF-IDF in parallel
        train_html['plaintext'] = parallel_apply(extract_plaintext, train_html['raw_html'], desc='Extracting plaintext (train)')
        test_html['plaintext'] = parallel_apply(extract_plaintext, test_html['raw_html'], desc='Extracting plaintext (test)')

        # Extract hyperlink-based features in parallel
        train_hyperlinks = parallel_apply(extract_hyperlink_stats, train_html['raw_html'], desc='Extracting hyperlink stats (train)')
        test_hyperlinks = parallel_apply(extract_hyperlink_stats, test_html['raw_html'], desc='Extracting hyperlink stats (test)')
        train_hyperlinks = pd.DataFrame(train_hyperlinks)
        test_hyperlinks = pd.DataFrame(test_hyperlinks)

        # --- Step 3: TF-IDF Vectorization of Plaintext ---
        # Remove empty documents
        train_html = train_html[train_html['plaintext'].str.strip().astype(bool)]
        train_html = train_html.reset_index(drop=True)

        print("Number of non-empty documents after cleaning:", len(train_html))
        print("Sample cleaned documents:", train_html['plaintext'][:5])

        if len(train_html) == 0:
            raise ValueError("All training documents are empty after cleaning. Please check your data extraction logic.")

        # Print unique words from sample documents
        sample_text = ' '.join(train_html['plaintext'][:10])
        words = set(sample_text.split())
        print("First 100 unique words in sample:", list(words)[:100])

        # Ensure all inputs are strings
        train_html['plaintext'] = train_html['plaintext'].astype(str)
        test_html['plaintext'] = test_html['plaintext'].astype(str)

        # Try fitting TF-IDF and catch errors (fix token_pattern)
        try:
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
            X_train_tfidf = tfidf.fit_transform(train_html['plaintext'])
            X_test_tfidf = tfidf.transform(test_html['plaintext'])
        except ValueError as e:
            print("TF-IDF with English stop words failed:", e)
            print("Trying again with stop_words=None ...")
            tfidf = TfidfVectorizer(max_features=1000, stop_words=None, token_pattern=r'(?u)\b\w+\b')
            X_train_tfidf = tfidf.fit_transform(train_html['plaintext'])
            X_test_tfidf = tfidf.transform(test_html['plaintext'])

        # Convert TF-IDF to DataFrame
        X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(X_train_tfidf.shape[1])])
        X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(X_test_tfidf.shape[1])])

        # --- Step 4: Combine All HTML Features ---
        X_train_html = pd.concat([train_hyperlinks.reset_index(drop=True), X_train_tfidf_df], axis=1)
        X_test_html = pd.concat([test_hyperlinks.reset_index(drop=True), X_test_tfidf_df], axis=1)

        print('Sample HTML features:')
        print(X_train_html.head())

        # Save features for next steps
        X_train_html.to_csv('notebooks/X_train_html_features.csv', index=False)
        X_test_html.to_csv('notebooks/X_test_html_features.csv', index=False)
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
