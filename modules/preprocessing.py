"""
Preprocessing Module
Normalisasi Min-Max dan Split Train-Test tanpa library
Pure Python implementation
"""

import random


def min_max_normalize(X):
    """
    Normalisasi data ke rentang [0, 1] menggunakan Min-Max scaling
    Formula: X_norm = (X - X_min) / (X_max - X_min)
    
    Args:
        X: List 2D berisi features
        
    Returns:
        list: Data yang sudah dinormalisasi
    """
    print("🔄 Normalizing data (Min-Max scaling)...")
    
    if not X or not X[0]:
        return X
    
    n_rows = len(X)
    n_cols = len(X[0])
    
    # Step 1: Hitung min dan max untuk setiap kolom
    min_vals = [float('inf')] * n_cols
    max_vals = [float('-inf')] * n_cols
    
    for row in X:
        for j in range(n_cols):
            if row[j] < min_vals[j]:
                min_vals[j] = row[j]
            if row[j] > max_vals[j]:
                max_vals[j] = row[j]
    
    # Step 2: Normalisasi setiap nilai
    X_normalized = []
    
    for row in X:
        normalized_row = []
        for j in range(n_cols):
            # Hindari division by zero
            if max_vals[j] - min_vals[j] == 0:
                normalized_row.append(0.0)
            else:
                normalized_value = (row[j] - min_vals[j]) / (max_vals[j] - min_vals[j])
                normalized_row.append(normalized_value)
        
        X_normalized.append(normalized_row)
    
    print(f"✅ Normalization completed: {n_rows} rows × {n_cols} columns")
    print(f"   Sample normalized values (first row): {[f'{val:.4f}' for val in X_normalized[0][:3]]}...")
    
    return X_normalized


def shuffle_data(X, y, seed=42):
    """
    Acak data dengan seed untuk reproducibility
    
    Args:
        X: Features
        y: Labels
        seed: Random seed untuk reproducibility
        
    Returns:
        tuple: (X_shuffled, y_shuffled)
    """
    print(f"🔀 Shuffling data (seed={seed})...")
    
    # Set random seed
    random.seed(seed)
    
    # Gabungkan X dan y
    combined = list(zip(X, y))
    
    # Shuffle
    random.shuffle(combined)
    
    # Pisahkan kembali
    X_shuffled = [item[0] for item in combined]
    y_shuffled = [item[1] for item in combined]
    
    print(f"✅ Data shuffled: {len(X_shuffled)} rows")
    
    return X_shuffled, y_shuffled


def split_train_test(X, y, test_ratio=0.2):
    """
    Split data menjadi train dan test set
    
    Args:
        X: Features
        y: Labels
        test_ratio: Proporsi data untuk testing (default 0.2 = 20%)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"✂️  Splitting data (test ratio={test_ratio})...")
    
    n_total = len(X)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test
    
    # Split index
    split_index = n_train
    
    # Split data
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    print(f"✅ Train set: {len(X_train)} rows ({n_train/n_total*100:.1f}%)")
    print(f"✅ Test set:  {len(X_test)} rows ({n_test/n_total*100:.1f}%)")
    print(f"   Train labels: 0={y_train.count(0)}, 1={y_train.count(1)}")
    print(f"   Test labels:  0={y_test.count(0)}, 1={y_test.count(1)}")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(X, y, test_ratio=0.2, seed=42):
    """
    Fungsi utama untuk preprocessing:
    1. Shuffle data
    2. Split train-test
    3. Normalize data
    
    Args:
        X: Features
        y: Labels
        test_ratio: Proporsi test data
        seed: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) yang sudah dinormalisasi
    """
    print("\n" + "="*60)
    print("🔧 PREPROCESSING DATA")
    print("="*60 + "\n")
    
    # Step 1: Shuffle
    X_shuffled, y_shuffled = shuffle_data(X, y, seed)
    
    # Step 2: Split
    X_train, X_test, y_train, y_test = split_train_test(X_shuffled, y_shuffled, test_ratio)
    
    # Step 3: Normalize (fit on train, transform both)
    # Hitung min-max dari train data saja untuk avoid data leakage
    n_cols = len(X_train[0])
    
    # Hitung min-max dari training data
    min_vals = [float('inf')] * n_cols
    max_vals = [float('-inf')] * n_cols
    
    for row in X_train:
        for j in range(n_cols):
            if row[j] < min_vals[j]:
                min_vals[j] = row[j]
            if row[j] > max_vals[j]:
                max_vals[j] = row[j]
    
    # Normalize train data
    X_train_norm = []
    for row in X_train:
        normalized_row = []
        for j in range(n_cols):
            if max_vals[j] - min_vals[j] == 0:
                normalized_row.append(0.0)
            else:
                normalized_value = (row[j] - min_vals[j]) / (max_vals[j] - min_vals[j])
                normalized_row.append(normalized_value)
        X_train_norm.append(normalized_row)
    
    # Normalize test data dengan min-max dari train
    X_test_norm = []
    for row in X_test:
        normalized_row = []
        for j in range(n_cols):
            if max_vals[j] - min_vals[j] == 0:
                normalized_row.append(0.0)
            else:
                normalized_value = (row[j] - min_vals[j]) / (max_vals[j] - min_vals[j])
                # Clip ke [0, 1] untuk handle outliers di test
                normalized_value = max(0.0, min(1.0, normalized_value))
                normalized_row.append(normalized_value)
        X_test_norm.append(normalized_row)
    
    print(f"✅ Normalization completed")
    print(f"   Train: {len(X_train_norm)} rows")
    print(f"   Test:  {len(X_test_norm)} rows")
    
    print("\n✅ Preprocessing completed!\n")
    
    return X_train_norm, X_test_norm, y_train, y_test


if __name__ == "__main__":
    # Test module
    print("Testing preprocessing module...")
    
    # Dummy data
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    y = [0, 1, 0, 1]
    
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_ratio=0.5)
    
    print("\nResults:")
    print("X_train:", X_train)
    print("X_test:", X_test)
    print("y_train:", y_train)
    print("y_test:", y_test)
