"""
Evaluation Module
Hitung Confusion Matrix dan Metrik Performa (Accuracy, Precision, Recall, F1-Score)
Pure Python implementation - sesuai requirement tugas
"""

import csv


def build_confusion_matrix(y_true, y_pred):
    """
    Buat confusion matrix dari hasil prediksi
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        
    Returns:
        dict: Confusion matrix dengan keys: TP, TN, FP, FN
    """
    tp = 0  # True Positive: Prediksi 1, Aktual 1
    tn = 0  # True Negative: Prediksi 0, Aktual 0
    fp = 0  # False Positive: Prediksi 1, Aktual 0 (Type I Error)
    fn = 0  # False Negative: Prediksi 0, Aktual 1 (Type II Error)
    
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
    
    confusion_matrix = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    
    return confusion_matrix


def calculate_accuracy(cm):
    """
    Hitung Accuracy dari confusion matrix
    Formula: (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        cm: Confusion matrix dictionary
        
    Returns:
        float: Accuracy (0-1)
    """
    tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    total = tp + tn + fp + fn
    
    if total == 0:
        return 0.0
    
    accuracy = (tp + tn) / total
    return accuracy


def calculate_precision(cm):
    """
    Hitung Precision dari confusion matrix
    Formula: TP / (TP + FP)
    Precision = Seberapa akurat prediksi positif
    
    Args:
        cm: Confusion matrix dictionary
        
    Returns:
        float: Precision (0-1)
    """
    tp, fp = cm['TP'], cm['FP']
    
    if (tp + fp) == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    return precision


def calculate_recall(cm):
    """
    Hitung Recall (Sensitivity) dari confusion matrix
    Formula: TP / (TP + FN)
    Recall = Seberapa banyak positif yang berhasil terdeteksi
    
    Args:
        cm: Confusion matrix dictionary
        
    Returns:
        float: Recall (0-1)
    """
    tp, fn = cm['TP'], cm['FN']
    
    if (tp + fn) == 0:
        return 0.0
    
    recall = tp / (tp + fn)
    return recall


def calculate_f1_score(precision, recall):
    """
    Hitung F1-Score dari Precision dan Recall
    Formula: 2 × (Precision × Recall) / (Precision + Recall)
    F1-Score = Harmonic mean dari Precision dan Recall
    
    Args:
        precision: Nilai precision
        recall: Nilai recall
        
    Returns:
        float: F1-Score (0-1)
    """
    if (precision + recall) == 0:
        return 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_all_metrics(y_true, y_pred):
    """
    Hitung semua metrik evaluasi sekaligus
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        
    Returns:
        dict: Dictionary berisi semua metrik
    """
    # Build confusion matrix
    cm = build_confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = calculate_accuracy(cm)
    precision = calculate_precision(cm)
    recall = calculate_recall(cm)
    f1_score = calculate_f1_score(precision, recall)
    
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    return metrics


def print_confusion_matrix(cm):
    """
    Print confusion matrix dalam format yang mudah dibaca
    
    Args:
        cm: Confusion matrix dictionary
    """
    print("\n" + "="*60)
    print("🎯 CONFUSION MATRIX")
    print("="*60 + "\n")
    
    print("                 Predicted")
    print("                 0        1")
    print("              ┌────────────────┐")
    print(f"Actual   0    │  {cm['TN']:>4}  │  {cm['FP']:>4}  │")
    print("              ├────────────────┤")
    print(f"         1    │  {cm['FN']:>4}  │  {cm['TP']:>4}  │")
    print("              └────────────────┘")
    
    print(f"\n  True Positive  (TP): {cm['TP']:>4}  - Correctly predicted as diabetes")
    print(f"  True Negative  (TN): {cm['TN']:>4}  - Correctly predicted as non-diabetes")
    print(f"  False Positive (FP): {cm['FP']:>4}  - Incorrectly predicted as diabetes")
    print(f"  False Negative (FN): {cm['FN']:>4}  - Incorrectly predicted as non-diabetes")
    print()


def print_metrics(metrics):
    """
    Print semua metrik evaluasi
    
    Args:
        metrics: Dictionary berisi semua metrik
    """
    print("="*60)
    print("📊 PERFORMANCE METRICS")
    print("="*60 + "\n")
    
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1_score']
    
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score  : {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    print("\n" + "="*60 + "\n")


def save_confusion_matrix_to_csv(cm, filepath):
    """
    Simpan confusion matrix ke CSV
    
    Args:
        cm: Confusion matrix dictionary
        filepath: Path file output
    """
    print(f"💾 Saving confusion matrix to {filepath}...")
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Header
            writer.writerow(['Metric', 'Value', 'Description'])
            
            # Data
            writer.writerow(['TP', cm['TP'], 'True Positive - Correctly predicted as diabetes'])
            writer.writerow(['TN', cm['TN'], 'True Negative - Correctly predicted as non-diabetes'])
            writer.writerow(['FP', cm['FP'], 'False Positive - Incorrectly predicted as diabetes'])
            writer.writerow(['FN', cm['FN'], 'False Negative - Incorrectly predicted as non-diabetes'])
        
        print(f"✅ Confusion matrix saved!")
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


def save_metrics_to_csv(metrics, filepath):
    """
    Simpan metrik evaluasi ke CSV
    
    Args:
        metrics: Dictionary berisi semua metrik
        filepath: Path file output
    """
    print(f"💾 Saving performance metrics to {filepath}...")
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Header
            writer.writerow(['Metric', 'Value', 'Percentage'])
            
            # Data
            writer.writerow(['Accuracy', f"{metrics['accuracy']:.4f}", f"{metrics['accuracy']*100:.2f}%"])
            writer.writerow(['Precision', f"{metrics['precision']:.4f}", f"{metrics['precision']*100:.2f}%"])
            writer.writerow(['Recall', f"{metrics['recall']:.4f}", f"{metrics['recall']*100:.2f}%"])
            writer.writerow(['F1-Score', f"{metrics['f1_score']:.4f}", f"{metrics['f1_score']*100:.2f}%"])
        
        print(f"✅ Metrics saved!")
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


def evaluate_and_save(y_true, y_pred, output_dir='results'):
    """
    Fungsi utama untuk evaluasi dan simpan hasil
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        output_dir: Direktori output untuk file CSV
        
    Returns:
        dict: Dictionary berisi semua metrik
    """
    print("\n" + "="*60)
    print("🔍 EVALUATING MODEL PERFORMANCE")
    print("="*60 + "\n")
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    
    # Print results
    print_confusion_matrix(metrics['confusion_matrix'])
    print_metrics(metrics)
    
    # Save to CSV
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
    metrics_path = os.path.join(output_dir, 'final_metrics.csv')
    
    save_confusion_matrix_to_csv(metrics['confusion_matrix'], cm_path)
    save_metrics_to_csv(metrics, metrics_path)
    
    print("\n✅ Evaluation completed!\n")
    
    return metrics


if __name__ == "__main__":
    # Test module
    print("Testing Evaluation module...")
    
    # Dummy data
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    
    metrics = evaluate_and_save(y_true, y_pred, output_dir='test_results')
    
    print("Test completed!")
