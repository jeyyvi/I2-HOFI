"""
Plot training history: loss, accuracy curves from saved history
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def plot_training_history(history_file, output_dir='./plots'):
    """
    Plot training and validation metrics over epochs
    
    Args:
        history_file: Path to training_history.json
        output_dir: Directory to save plots
    """
    
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = history['epoch']
    
    # 1. Training and Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training and Validation Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    if 'val_acc' in history and history['val_acc']:
        axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    axes[1, 0].plot(epochs, history['lr'], 'g-^', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Accuracy Gap (Overfitting indicator)
    if 'val_acc' in history and history['val_acc']:
        acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
        axes[1, 1].plot(epochs, acc_gap, 'purple', linewidth=2, marker='o', markersize=6)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].fill_between(epochs, 0, acc_gap, where=[g > 0 for g in acc_gap], 
                                color='red', alpha=0.3, label='Overfitting')
        axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Train Acc - Val Acc', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Accuracy Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Validation data not available', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'✓ Training history plot saved to: {plot_path}')
    plt.close()
    
    # Create summary statistics
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Epochs: {len(epochs)}\n")
        f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}\n")
        
        if 'val_loss' in history and history['val_loss']:
            f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}\n")
            f.write(f"Best Validation Accuracy: {max(history['val_acc']):.4f} (Epoch {history['epoch'][history['val_acc'].index(max(history['val_acc']))]})\n")
        
        f.write(f"Final Learning Rate: {history['lr'][-1]:.6f}\n")
        
        if 'val_acc' in history and history['val_acc']:
            final_gap = history['train_acc'][-1] - history['val_acc'][-1]
            f.write(f"\nFinal Accuracy Gap: {final_gap:.4f}\n")
            if final_gap > 0.1:
                f.write("⚠ Warning: Large gap suggests overfitting\n")
    
    print(f'✓ Training summary saved to: {summary_path}')
    
    # Print summary to console
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total Epochs: {len(epochs)}")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    if 'val_acc' in history and history['val_acc']:
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
        print(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_history.py <path_to_training_history.json>")
        print("Example: python plot_training_history.py TrainedModels/i2hofi/training_history.json")
        sys.exit(1)
    
    history_file = sys.argv[1]
    
    if not os.path.exists(history_file):
        print(f"Error: History file not found: {history_file}")
        sys.exit(1)
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './training_plots'
    
    plot_training_history(history_file, output_dir)


if __name__ == "__main__":
    main()