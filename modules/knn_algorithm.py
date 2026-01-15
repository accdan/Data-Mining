"""
KNN Algorithm Module
Implementasi K-Nearest Neighbors dari nol tanpa library
Pure Python dengan perhitungan Euclidean Distance
"""

import math


def euclidean_distance(vec1, vec2):
    """
    Hitung jarak Euclidean antara dua vektor
    Formula: sqrt(Σ(x1 - x2)²)
    
    Args:
        vec1: Vektor pertama
        vec2: Vektor kedua
        
    Returns:
        float: Jarak Euclidean
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    # Hitung sum of squared differences
    squared_diff_sum = 0
    for i in range(len(vec1)):
        diff = vec1[i] - vec2[i]
        squared_diff_sum += diff * diff
    
    # Return square root
    return math.sqrt(squared_diff_sum)


def get_k_neighbors(X_train, y_train, test_row, k):
    """
    Cari k tetangga terdekat dari test_row di training data
    
    Args:
        X_train: Training features
        y_train: Training labels
        test_row: Data test yang ingin diprediksi
        k: Jumlah tetangga yang dicari
        
    Returns:
        list: List of (distance, label) untuk k tetangga terdekat
    """
    distances = []
    
    # Hitung jarak ke semua training data
    for i in range(len(X_train)):
        dist = euclidean_distance(test_row, X_train[i])
        distances.append((dist, y_train[i]))
    
    # Sort berdasarkan jarak (ascending)
    distances.sort(key=lambda x: x[0])
    
    # Ambil k tetangga terdekat
    neighbors = distances[:k]
    
    return neighbors


def majority_vote(neighbors):
    """
    Voting mayoritas dari tetangga untuk klasifikasi
    
    Args:
        neighbors: List of (distance, label)
        
    Returns:
        int: Label hasil voting (0 atau 1)
    """
    # Count votes untuk setiap kelas
    votes = {}
    
    for distance, label in neighbors:
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    
    # Kembalikan label dengan vote terbanyak
    max_votes = 0
    predicted_label = 0
    
    for label, count in votes.items():
        if count > max_votes:
            max_votes = count
            predicted_label = label
    
    return predicted_label


def knn_predict_single(X_train, y_train, test_row, k):
    """
    Prediksi untuk satu test row
    
    Args:
        X_train: Training features
        y_train: Training labels
        test_row: Satu baris data test
        k: Jumlah tetangga
        
    Returns:
        int: Predicted label
    """
    neighbors = get_k_neighbors(X_train, y_train, test_row, k)
    prediction = majority_vote(neighbors)
    return prediction


def knn_predict(X_train, y_train, X_test, k, verbose=False):
    """
    Prediksi untuk semua test data
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        k: Jumlah tetangga
        verbose: Print progress atau tidak
        
    Returns:
        list: Predictions untuk semua test data
    """
    predictions = []
    
    total = len(X_test)
    
    for i, test_row in enumerate(X_test):
        prediction = knn_predict_single(X_train, y_train, test_row, k)
        predictions.append(prediction)
        
        # Print progress
        if verbose and (i + 1) % 20 == 0:
            print(f"   Predicted {i + 1}/{total} samples...")
    
    if verbose:
        print(f"✅ Prediction completed: {total} samples")
    
    return predictions


def calculate_accuracy(y_true, y_pred):
    """
    Hitung akurasi prediksi
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        
    Returns:
        float: Akurasi (0-1)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    
    accuracy = correct / len(y_true)
    return accuracy


def knn_classify(X_train, y_train, X_test, y_test, k, verbose=True):
    """
    Fungsi utama untuk klasifikasi KNN dengan evaluasi
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        k: Jumlah tetangga
        verbose: Print detail atau tidak
        
    Returns:
        tuple: (predictions, accuracy)
    """
    if verbose:
        print(f"\n🤖 Running KNN Classification (k={k})...")
    
    # Prediksi
    predictions = knn_predict(X_train, y_train, X_test, k, verbose)
    
    # Hitung akurasi
    accuracy = calculate_accuracy(y_test, predictions)
    
    if verbose:
        print(f"✅ Accuracy: {accuracy*100:.2f}%")
    
    return predictions, accuracy


if __name__ == "__main__":
    # Test module
    print("Testing KNN Algorithm...")
    
    # Dummy data
    X_train = [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.1, 0.1],
        [0.9, 0.9]
    ]
    y_train = [0, 1, 0, 1]
    
    X_test = [
        [0.05, 0.05],
        [0.95, 0.95]
    ]
    y_test = [0, 1]
    
    # Test dengan k=3
    predictions, accuracy = knn_classify(X_train, y_train, X_test, y_test, k=3)
    
    print(f"\nResults:")
    print(f"Predictions: {predictions}")
    print(f"True labels: {y_test}")
    print(f"Accuracy: {accuracy*100:.2f}%")
