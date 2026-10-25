import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold

def main():
    print("1. Loading and Shuffling dataset...")
    # Added .sample(frac=1) to randomize the row order
    df = pd.read_csv('data/processed/resume_jd_clean.csv').sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("2. Generating Embeddings (This will take ~60 seconds)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    resume_emb = embedder.encode(df['resume_text'].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    jd_emb = embedder.encode(df['jd_text'].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    
    print("3. Engineering Features...")
    dot_product = np.sum(resume_emb * jd_emb, axis=1, keepdims=True)
    l2_distance = np.linalg.norm(resume_emb - jd_emb, axis=1, keepdims=True)
    abs_diff = np.abs(resume_emb - jd_emb)
    hadamard = resume_emb * jd_emb
    
    X = np.hstack([resume_emb, jd_emb, abs_diff, hadamard, dot_product, l2_distance]).astype(np.float32)
    y = df['label'].to_numpy().astype(int)
    
    print("4. Calculating Learning Curve across 3 Folds (Please wait)...")
    clf = HistGradientBoostingClassifier(
        l2_regularization=0.01, 
        max_depth=None, 
        max_iter=200, 
        max_leaf_nodes=31, 
        min_samples_leaf=20,
        random_state=42
    )
    
    # Use StratifiedKFold to guarantee both classes are in every data slice
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, scoring='f1', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    print("5. Generating Professional Plot...")
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training F1-Score', color='#8b5cf6', marker='o', linewidth=2)
    plt.plot(train_sizes, test_mean, label='Cross-Validation F1-Score', color='#22c55e', marker='s', linewidth=2)
    
    plt.title('HistGradientBoosting Learning Curve (F1-Score)', fontsize=14, pad=15)
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Success! The graph has been saved as 'learning_curve.png' in your folder.")

if __name__ == "__main__":
    main()