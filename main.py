"""
MAIN PROGRAM
Klasifikasi Pima Indians Diabetes menggunakan KNN dan Metode Elbow
Tanpa library pengolah data (Pure Python + Kagglehub + Matplotlib)

Tugas: Data Mining - Klasifikasi
Author: Mahasiswa Informatika
Date: 2026-01-15
"""

import os
import sys

# Import custom modules
from modules import data_loader
from modules import preprocessing
from modules import knn_algorithm
from modules import elbow_method
from modules import evaluation
from modules import visualization


def print_header():
    """Print header program"""
    print("\n" + "="*70)
    print("║" + " "*68 + "║")
    print("║" + "  K-NEAREST NEIGHBORS CLASSIFICATION".center(68) + "║")
    print("║" + "  Pima Indians Diabetes Dataset".center(68) + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Metode: KNN + Elbow Method".center(68) + "║")
    print("║" + "  Pure Python Implementation (No Pandas/Scikit-learn)".center(68) + "║")
    print("║" + " "*68 + "║")
    print("="*70 + "\n")


def print_section(title):
    """Print section separator"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    """
    Fungsi utama program
    
    Alur eksekusi:
    1. Load data dari Kaggle
    2. Preprocessing (normalize + split)
    3. Elbow Analysis (k=1 to 15)
    4. Visualisasi grafik Elbow
    5. Pilih k terbaik
    6. Train KNN final dengan k terbaik
    7. Evaluasi & tampilkan metrik
    8. Export semua hasil ke CSV
    """
    
    # Print header
    print_header()
    
    # ========== STEP 1: LOAD DATA ==========
    print_section("STEP 1: LOADING DATA FROM KAGGLE")
    
    try:
        X, y = data_loader.load_and_prepare_data()
        print(f"✅ Data loaded successfully!")
        print(f"   Total samples: {len(X)}")
        print(f"   Features: {len(X[0])}")
        print(f"   Class 0 (Non-diabetes): {y.count(0)} samples")
        print(f"   Class 1 (Diabetes): {y.count(1)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # ========== STEP 2: PREPROCESSING ==========
    print_section("STEP 2: PREPROCESSING DATA")
    
    try:
        X_train, X_test, y_train, y_test = preprocessing.preprocess_data(
            X, y, test_ratio=0.2, seed=42
        )
        print(f"✅ Preprocessing completed!")
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        sys.exit(1)
    
    # ========== STEP 3: ELBOW ANALYSIS ==========
    print_section("STEP 3: ELBOW METHOD ANALYSIS")
    
    try:
        # Range k yang akan ditest (1-15 sesuai requirement "minimal range")
        k_range = range(1, 16)
        
        print(f"🔍 Testing k values from {min(k_range)} to {max(k_range)}...")
        print(f"⏳ This may take a few minutes...\n")
        
        # Jalankan analisis Elbow
        elbow_results = elbow_method.run_elbow_analysis(
            X_train, y_train, X_test, y_test, k_range
        )
        
        # Tampilkan hasil dalam tabel
        elbow_method.print_results_table(elbow_results)
        
        # Simpan hasil ke CSV
        elbow_method.save_elbow_to_csv(elbow_results, 'results/elbow_analysis.csv')
        
    except Exception as e:
        print(f"❌ Error in Elbow analysis: {e}")
        sys.exit(1)
    
    # ========== STEP 4: VISUALISASI GRAFIK ELBOW ==========
    print_section("STEP 4: CREATING ELBOW CURVE VISUALIZATION")
    
    try:
        visualization.visualize_elbow_results(elbow_results, output_dir='results')
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        # Continue walaupun visualisasi gagal
    
    # ========== STEP 5: PILIH K TERBAIK ==========
    print_section("STEP 5: SELECTING OPTIMAL K")
    
    # Cari k dengan error rate terendah
    best_k_result = min(elbow_results, key=lambda x: x[1])
    best_k = best_k_result[0]
    best_error = best_k_result[1]
    best_accuracy = best_k_result[2]
    
    print(f"🏆 OPTIMAL K SELECTED")
    print(f"   k = {best_k}")
    print(f"   Error Rate = {best_error:.2f}%")
    print(f"   Accuracy = {best_accuracy*100:.2f}%")
    
    # User dapat mengubah k secara manual jika mau
    print(f"\n💡 You can also manually select k from the graph if needed.")
    
    # ========== STEP 6: TRAIN FINAL MODEL ==========
    print_section(f"STEP 6: TRAINING FINAL MODEL (k={best_k})")
    
    try:
        print(f"🤖 Running KNN classification with k={best_k}...")
        
        final_predictions, final_accuracy = knn_algorithm.knn_classify(
            X_train, y_train, X_test, y_test, k=best_k, verbose=True
        )
        
        print(f"\n✅ Final model trained successfully!")
        
    except Exception as e:
        print(f"❌ Error training final model: {e}")
        sys.exit(1)
    
    # ========== STEP 7: EVALUASI PERFORMA ==========
    print_section("STEP 7: EVALUATING MODEL PERFORMANCE")
    
    try:
        # Hitung semua metrik
        final_metrics = evaluation.evaluate_and_save(
            y_test, final_predictions, output_dir='results'
        )
        
        # Tampilkan summary
        print("\n📊 FINAL RESULTS SUMMARY")
        print("="*70)
        print(f"  Best K Value: {best_k}")
        print(f"  Accuracy:     {final_metrics['accuracy']*100:.2f}%")
        print(f"  Precision:    {final_metrics['precision']*100:.2f}%")
        print(f"  Recall:       {final_metrics['recall']*100:.2f}%")
        print(f"  F1-Score:     {final_metrics['f1_score']*100:.2f}%")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        sys.exit(1)
    
    # ========== STEP 8: OUTPUT FILES ==========
    print_section("STEP 8: OUTPUT FILES GENERATED")
    
    print("📁 All results have been saved to the 'results/' directory:\n")
    print("   ✅ elbow_analysis.csv      - Elbow analysis results for all k values")
    print("   ✅ confusion_matrix.csv    - Confusion matrix (TP, TN, FP, FN)")
    print("   ✅ final_metrics.csv       - Performance metrics (Acc, Prec, Rec, F1)")
    print("   ✅ elbow_curve.png         - Elbow curve visualization (detailed)")
    print("   ✅ elbow_curve_simple.png  - Elbow curve visualization (simple)")
    
    # ========== COMPLETION ==========
    print("\n" + "="*70)
    print("║" + " "*68 + "║")
    print("║" + "  ✅ PROGRAM COMPLETED SUCCESSFULLY!".center(68) + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Semua hasil telah disimpan di folder 'results/'".center(68) + "║")
    print("║" + "  Silakan periksa file CSV dan grafik untuk analisis".center(68) + "║")
    print("║" + " "*68 + "║")
    print("="*70 + "\n")
    
    print("💡 TIP: Gunakan grafik Elbow untuk memahami trade-off antara")
    print("        kompleksitas model (k) dan performa (accuracy/error rate)\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
