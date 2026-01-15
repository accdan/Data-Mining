"""
Elbow Method Module
Analisis untuk mencari nilai k optimal menggunakan metode Elbow
Pure Python implementation
"""

from . import knn_algorithm
import csv


def calculate_error_rate(y_true, y_pred):
    """
    Hitung error rate (persentase prediksi salah)
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        
    Returns:
        float: Error rate dalam persentase (0-100)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    errors = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors += 1
    
    error_rate = (errors / len(y_true)) * 100
    return error_rate


def run_elbow_analysis(X_train, y_train, X_test, y_test, k_range=None):
    """
    Jalankan analisis Elbow untuk range nilai k
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        k_range: Range nilai k yang akan ditest (default: 1-15)
        
    Returns:
        list: List of tuples (k, error_rate, accuracy)
    """
    if k_range is None:
        k_range = range(1, 16)  # Default: k=1 sampai k=15
    
    print("\n" + "="*60)
    print("📈 ELBOW METHOD ANALYSIS")
    print("="*60 + "\n")
    
    print(f"Testing k values from {min(k_range)} to {max(k_range)}...")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")
    
    results = []
    
    for k in k_range:
        print(f"🔍 Testing k={k}...")
        
        # Jalankan KNN
        predictions = knn_algorithm.knn_predict(X_train, y_train, X_test, k, verbose=False)
        
        # Hitung akurasi dan error rate
        accuracy = knn_algorithm.calculate_accuracy(y_test, predictions)
        error_rate = calculate_error_rate(y_test, predictions)
        
        # Simpan hasil
        results.append((k, error_rate, accuracy))
        
        print(f"   Accuracy: {accuracy*100:.2f}% | Error Rate: {error_rate:.2f}%")
    
    print("\n✅ Elbow analysis completed!\n")
    
    return results


def find_elbow_point(results):
    """
    Cari elbow point dari hasil analisis
    Metode: Cari titik dimana penurunan error rate mulai melambat
    
    Args:
        results: List of (k, error_rate, accuracy)
        
    Returns:
        int: Nilai k optimal (elbow point)
    """
    if len(results) < 3:
        return results[0][0]  # Return k pertama jika data terlalu sedikit
    
    # Hitung rate of change (penurunan error rate)
    changes = []
    for i in range(len(results) - 1):
        k1, error1, _ = results[i]
        k2, error2, _ = results[i + 1]
        change = abs(error2 - error1)
        changes.append((k2, change))
    
    # Cari titik dimana perubahan mulai kecil (< 2%)
    threshold = 2.0
    for k, change in changes:
        if change < threshold:
            return k
    
    # Jika tidak ditemukan, return k dengan error rate terendah
    min_error_k = min(results, key=lambda x: x[1])[0]
    return min_error_k


def print_results_table(results):
    """
    Print hasil dalam bentuk tabel ASCII
    
    Args:
        results: List of (k, error_rate, accuracy)
    """
    print("\n" + "="*60)
    print("📋 ELBOW ANALYSIS RESULTS")
    print("="*60 + "\n")
    
    # Header
    print(f"{'k':<5} {'Error Rate':<15} {'Accuracy':<15}")
    print("-" * 40)
    
    # Data rows
    for k, error_rate, accuracy in results:
        print(f"{k:<5} {error_rate:>6.2f}%{'':<8} {accuracy*100:>6.2f}%")
    
    # Best k
    best_k = min(results, key=lambda x: x[1])
    print("\n" + "="*60)
    print(f"🏆 Best k: {best_k[0]} (Error Rate: {best_k[1]:.2f}%, Accuracy: {best_k[2]*100:.2f}%)")
    print("="*60 + "\n")


def save_elbow_to_csv(results, filepath):
    """
    Simpan hasil analisis Elbow ke CSV
    
    Args:
        results: List of (k, error_rate, accuracy)
        filepath: Path file output CSV
    """
    print(f"💾 Saving Elbow analysis to {filepath}...")
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Header
            writer.writerow(['k', 'error_rate', 'accuracy'])
            
            # Data
            for k, error_rate, accuracy in results:
                writer.writerow([k, f"{error_rate:.2f}", f"{accuracy*100:.2f}"])
        
        print(f"✅ Results saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


if __name__ == "__main__":
    # Test module
    print("Testing Elbow Method module...")
    
    # Dummy results
    results = [
        (1, 35.0, 0.65),
        (3, 28.0, 0.72),
        (5, 25.0, 0.75),
        (7, 24.0, 0.76),
        (9, 23.5, 0.765),
    ]
    
    print_results_table(results)
    
    elbow_k = find_elbow_point(results)
    print(f"Detected elbow point: k={elbow_k}")
