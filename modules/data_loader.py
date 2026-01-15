"""
Data Loader Module
Download dan load dataset Pima Indians Diabetes dari Kaggle
Tanpa menggunakan Pandas - pure Python dengan modul csv
"""

import csv
import os


def download_pima_dataset():
    """
    Download dataset Pima Indians Diabetes dari Kaggle menggunakan kagglehub
    
    Returns:
        str: Path ke file CSV dataset
    """
    print("📥 Downloading Pima Indians Diabetes dataset from Kaggle...")
    
    try:
        import kagglehub
        
        # Download dataset dari Kaggle
        # Dataset: UCI ML Repository - Pima Indians Diabetes
        path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
        
        # Cari file diabetes.csv di dalam folder yang didownload
        csv_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    break
            if csv_file:
                break
        
        if csv_file:
            print(f"✅ Dataset downloaded successfully: {csv_file}")
            return csv_file
        else:
            raise FileNotFoundError("CSV file not found in downloaded dataset")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise


def load_csv_to_list(filepath):
    """
    Membaca file CSV dan convert ke list of lists
    
    Args:
        filepath: Path ke file CSV
        
    Returns:
        list: Data dalam bentuk list 2D
    """
    print(f"📂 Loading CSV file: {filepath}")
    
    data = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip header
            header = next(csv_reader)
            print(f"📋 Columns: {', '.join(header)}")
            
            # Baca semua baris
            for row in csv_reader:
                if row:  # Skip empty rows
                    data.append(row)
        
        print(f"✅ Loaded {len(data)} rows")
        return data
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        raise


def convert_to_float(data):
    """
    Konversi semua string di data menjadi float
    
    Args:
        data: List 2D berisi string
        
    Returns:
        list: List 2D berisi float
    """
    print("🔄 Converting data to float...")
    
    converted_data = []
    
    for row in data:
        converted_row = []
        for value in row:
            try:
                # Konversi ke float
                converted_row.append(float(value))
            except ValueError:
                # Jika gagal convert, skip atau set default
                print(f"⚠️  Warning: Cannot convert '{value}' to float, using 0.0")
                converted_row.append(0.0)
        
        converted_data.append(converted_row)
    
    print(f"✅ Converted {len(converted_data)} rows to float")
    return converted_data


def separate_features_labels(data):
    """
    Pisahkan features (X) dan labels (y) dari dataset
    Dataset Pima: 8 features + 1 label (kolom terakhir)
    
    Args:
        data: List 2D dengan features dan label
        
    Returns:
        tuple: (X, y) dimana X adalah features dan y adalah labels
    """
    print("✂️  Separating features and labels...")
    
    X = []  # Features
    y = []  # Labels
    
    for row in data:
        # Kolom 0-7: Features
        # Kolom 8: Label (Outcome)
        X.append(row[:-1])  # Semua kecuali kolom terakhir
        y.append(int(row[-1]))  # Kolom terakhir sebagai label
    
    print(f"✅ Features: {len(X)} rows × {len(X[0])} columns")
    print(f"✅ Labels: {len(y)} rows")
    print(f"✅ Class distribution: 0={y.count(0)}, 1={y.count(1)}")
    
    return X, y


def load_and_prepare_data():
    """
    Fungsi utama untuk load dan prepare data
    
    Returns:
        tuple: (X, y) features dan labels yang sudah siap diproses
    """
    print("\n" + "="*60)
    print("🚀 LOADING AND PREPARING DATA")
    print("="*60 + "\n")
    
    # Step 1: Download dataset
    csv_path = download_pima_dataset()
    
    # Step 2: Load CSV ke list
    raw_data = load_csv_to_list(csv_path)
    
    # Step 3: Convert ke float
    numeric_data = convert_to_float(raw_data)
    
    # Step 4: Pisahkan features dan labels
    X, y = separate_features_labels(numeric_data)
    
    print("\n✅ Data preparation completed!\n")
    
    return X, y


if __name__ == "__main__":
    # Test module
    X, y = load_and_prepare_data()
    print(f"Sample data (first 3 rows):")
    for i in range(min(3, len(X))):
        print(f"  X[{i}]: {X[i]}")
        print(f"  y[{i}]: {y[i]}")
