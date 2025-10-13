"""
Visualization of Model Training Results
Generates comprehensive visualizations: accuracy, loss, confusion matrix, etc.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')
OUTPUT_DIR = RESULTS_DIR / 'visualizations'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_training_history():
    """Load training history from JSON"""
    history_files = sorted(MODELS_DIR.glob('training_history_*.json'))
    
    if not history_files:
        print("No training history found")
        return None
    
    latest_history = history_files[-1]
    
    with open(latest_history, 'r') as f:
        history = json.load(f)
    
    print(f"Loaded history from: {latest_history}")
    return history

def plot_training_metrics(history):
    """Plot training and validation accuracy/loss"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1 = axes[0]
    epochs = range(1, len(history['accuracy']) + 1)
    
    ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add final accuracy values
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    ax1.text(0.02, 0.98, f'Final Training: {final_train_acc:.4f}\nFinal Validation: {final_val_acc:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Loss
    ax2 = axes[1]
    ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add final loss values
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    ax2.text(0.02, 0.98, f'Final Training: {final_train_loss:.4f}\nFinal Validation: {final_val_loss:.4f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'training_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()

def plot_confusion_matrix():
    """Plot confusion matrix from evaluation results"""
    # Check if confusion matrix exists
    confusion_files = sorted(RESULTS_DIR.glob('confusion_matrix_*.npy'))
    
    if not confusion_files:
        print("No confusion matrix found. Run model evaluation first.")
        return
    
    # Load latest confusion matrix
    latest_cm = confusion_files[-1]
    cm = np.load(latest_cm)
    
    # Load class names
    class_files = sorted(MODELS_DIR.glob('class_names_*.json'))
    if class_files:
        with open(class_files[-1], 'r') as f:
            class_names = json.load(f)
    else:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()
    
    # Also plot non-normalized version
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'confusion_matrix_counts.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()

def plot_class_distribution():
    """Plot distribution of images across classes"""
    # Try to load from training data
    data_dir = Path('data/processed/train')
    
    if not data_dir.exists():
        print("Training data not found. Skipping class distribution plot.")
        return
    
    # Count images per class
    class_counts = {}
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            class_counts[class_dir.name] = count
    
    if not class_counts:
        print("No classes found in training data.")
        return
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(classes, counts, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Color the bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title('Training Data Distribution by Class', fontsize=16, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add total
    total = sum(counts)
    ax.text(0.98, 0.98, f'Total Images: {total:,}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'class_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()

def plot_per_class_accuracy():
    """Plot accuracy per class from confusion matrix"""
    confusion_files = sorted(RESULTS_DIR.glob('confusion_matrix_*.npy'))
    
    if not confusion_files:
        print("No confusion matrix found. Skipping per-class accuracy plot.")
        return
    
    cm = np.load(confusion_files[-1])
    
    # Load class names
    class_files = sorted(MODELS_DIR.glob('class_names_*.json'))
    if class_files:
        with open(class_files[-1], 'r') as f:
            class_names = json.load(f)
    else:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(len(class_names)):
        if cm[i].sum() > 0:
            accuracy = cm[i, i] / cm[i].sum()
            class_accuracies.append((class_names[i], accuracy))
        else:
            class_accuracies.append((class_names[i], 0))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    classes, accuracies = zip(*class_accuracies)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if acc >= 0.9 else 'orange' if acc >= 0.8 else 'red' for acc in accuracies]
    bars = ax.barh(classes, accuracies, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_xlim(0, 1.0)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{acc:.2%}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_accuracy = np.mean(accuracies)
    ax.axvline(avg_accuracy, color='blue', linestyle='--', linewidth=2, label=f'Average: {avg_accuracy:.2%}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'per_class_accuracy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations"""
    print("=" * 60)
    print("GENERATING TRAINING VISUALIZATIONS")
    print("=" * 60)
    print()
    
    # Training metrics
    print("1. Training Metrics...")
    history = load_training_history()
    if history:
        plot_training_metrics(history)
    
    # Confusion matrix
    print("\n2. Confusion Matrix...")
    plot_confusion_matrix()
    
    # Class distribution
    print("\n3. Class Distribution...")
    plot_class_distribution()
    
    # Per-class accuracy
    print("\n4. Per-Class Accuracy...")
    plot_per_class_accuracy()
    
    print()
    print("=" * 60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_visualizations()