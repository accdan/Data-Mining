"""
Visualization Module
Visualisasi grafik Elbow menggunakan Matplotlib
"""

import matplotlib.pyplot as plt
import os


def plot_elbow_curve(results, save_path='results/elbow_curve.png', show_plot=False):
    """
    Plot grafik Elbow untuk analisis k optimal
    
    Args:
        results: List of tuples (k, error_rate, accuracy)
        save_path: Path untuk menyimpan gambar
        show_plot: Tampilkan plot atau tidak (default False untuk headless)
    """
    print(f"\n📊 Creating Elbow curve visualization...")
    
    # Extract data
    k_values = [r[0] for r in results]
    error_rates = [r[1] for r in results]
    accuracies = [r[2] * 100 for r in results]  # Convert to percentage
    
    # Create figure dengan 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== SUBPLOT 1: Error Rate vs K ==========
    ax1.plot(k_values, error_rates, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Method: Error Rate vs K', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(k_values)
    
    # Highlight best k (lowest error rate)
    best_idx = error_rates.index(min(error_rates))
    best_k = k_values[best_idx]
    best_error = error_rates[best_idx]
    
    ax1.plot(best_k, best_error, marker='*', markersize=20, color='#2ecc71', 
             label=f'Best k={best_k} (Error={best_error:.2f}%)', zorder=5)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Annotate best point
    ax1.annotate(f'k={best_k}\n{best_error:.2f}%', 
                 xy=(best_k, best_error), 
                 xytext=(best_k + 1, best_error + 2),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    
    # ========== SUBPLOT 2: Accuracy vs K ==========
    ax2.plot(k_values, accuracies, marker='s', linewidth=2, markersize=8, color='#3498db')
    ax2.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Elbow Method: Accuracy vs K', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(k_values)
    
    # Highlight best k (highest accuracy)
    best_accuracy = accuracies[best_idx]
    
    ax2.plot(best_k, best_accuracy, marker='*', markersize=20, color='#2ecc71',
             label=f'Best k={best_k} (Acc={best_accuracy:.2f}%)', zorder=5)
    ax2.legend(fontsize=10, loc='lower right')
    
    # Annotate best point
    ax2.annotate(f'k={best_k}\n{best_accuracy:.2f}%', 
                 xy=(best_k, best_accuracy), 
                 xytext=(best_k + 1, best_accuracy - 2),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    
    # Overall title
    fig.suptitle('K-Nearest Neighbors: Elbow Method Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Elbow curve saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_simple_elbow(results, save_path='results/elbow_curve_simple.png'):
    """
    Plot grafik Elbow sederhana (hanya error rate)
    
    Args:
        results: List of tuples (k, error_rate, accuracy)
        save_path: Path untuk menyimpan gambar
    """
    print(f"\n📊 Creating simple Elbow curve...")
    
    # Extract data
    k_values = [r[0] for r in results]
    error_rates = [r[1] for r in results]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot
    plt.plot(k_values, error_rates, marker='o', linewidth=2.5, markersize=10, 
             color='#e74c3c', label='Error Rate')
    
    # Highlight best k
    best_idx = error_rates.index(min(error_rates))
    best_k = k_values[best_idx]
    best_error = error_rates[best_idx]
    
    plt.plot(best_k, best_error, marker='*', markersize=25, color='#2ecc71', 
             label=f'Optimal k={best_k}', zorder=5)
    
    # Labels and title
    plt.xlabel('K (Number of Neighbors)', fontsize=13, fontweight='bold')
    plt.ylabel('Error Rate (%)', fontsize=13, fontweight='bold')
    plt.title('Elbow Method: Finding Optimal K', fontsize=15, fontweight='bold', pad=20)
    
    # Grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(k_values)
    plt.legend(fontsize=11, loc='upper right')
    
    # Annotate best point
    plt.annotate(f'Best k={best_k}\nError={best_error:.2f}%', 
                 xy=(best_k, best_error), 
                 xytext=(best_k + 2, best_error + 3),
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Save
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Simple Elbow curve saved to: {save_path}")
    plt.close()


def visualize_elbow_results(results, output_dir='results'):
    """
    Fungsi utama untuk membuat semua visualisasi
    
    Args:
        results: List of tuples (k, error_rate, accuracy)
        output_dir: Direktori output
    """
    print("\n" + "="*60)
    print("🎨 CREATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Create both plots
    plot_elbow_curve(results, save_path=f'{output_dir}/elbow_curve.png')
    plot_simple_elbow(results, save_path=f'{output_dir}/elbow_curve_simple.png')
    
    print("\n✅ All visualizations created!\n")


if __name__ == "__main__":
    # Test module
    print("Testing Visualization module...")
    
    # Dummy results
    results = [
        (1, 35.71, 0.6429),
        (3, 28.57, 0.7143),
        (5, 25.97, 0.7403),
        (7, 24.03, 0.7597),
        (9, 23.38, 0.7662),
        (11, 23.38, 0.7662),
        (13, 24.03, 0.7597),
        (15, 24.68, 0.7532),
    ]
    
    visualize_elbow_results(results, output_dir='test_results')
    
    print("Test completed!")
