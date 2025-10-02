"""
Test the trained model and visualize predictions
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# Disable warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths
PROCESSED_DIR = os.path.join('data', 'processed')
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Find the latest model
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
if not model_files:
    print("❌ No model found! Please train the model first.")
    exit()

MODEL_PATH = os.path.join(MODELS_DIR, sorted(model_files)[-1])

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_model_and_classes():
    """Load trained model and class names"""
    
    print("\n" + "=" * 50)
    print("STEP 1: LOADING MODEL")
    print("=" * 50)
    
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    print(f"\n✅ Model loaded from: {MODEL_PATH}")
    
    # Load class names
    classes_path = os.path.join(MODELS_DIR, 'class_names.json')
    with open(classes_path, 'r') as f:
        class_names = json.load(f)
    
    print(f"✅ Classes loaded: {len(class_names)} categories")
    print(f"   {class_names}")
    
    return model, class_names

def load_test_data(class_names):
    """Load test dataset"""
    
    print("\n" + "=" * 50)
    print("STEP 2: LOADING TEST DATA")
    print("=" * 50)
    
    test_ds = keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✅ Test dataset loaded")
    
    return test_ds

def evaluate_model(model, test_ds, class_names):
    """Evaluate model on test set"""
    
    print("\n" + "=" * 50)
    print("STEP 3: EVALUATING MODEL")
    print("=" * 50)
    
    print("\n🧪 Testing model on unseen data...")
    
    # Evaluate
    results = model.evaluate(test_ds, verbose=1)
    
    print("\n📊 Test Results:")
    print(f"   Test Loss:     {results[0]:.4f}")
    print(f"   Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    
    return results

def create_confusion_matrix(model, test_ds, class_names):
    """Create confusion matrix"""
    
    print("\n" + "=" * 50)
    print("STEP 4: CREATING CONFUSION MATRIX")
    print("=" * 50)
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {output_path}")
    
    plt.show()
    print("📊 Close the window to continue...")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def visualize_predictions(model, test_ds, class_names, num_samples=9):
    """Visualize sample predictions"""
    
    print("\n" + "=" * 50)
    print("STEP 5: VISUALIZING PREDICTIONS")
    print("=" * 50)
    
    # Get one batch
    for images, labels in test_ds.take(1):
        predictions = model.predict(images, verbose=0)
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        
        for idx in range(min(num_samples, len(images))):
            i = idx // 3
            j = idx % 3
            
            # Get image
            img = images[idx].numpy().astype("uint8")
            
            # Get true and predicted labels
            true_label = class_names[np.argmax(labels[idx])]
            pred_label = class_names[np.argmax(predictions[idx])]
            confidence = np.max(predictions[idx]) * 100
            
            # Plot
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            # Title with color
            color = 'green' if true_label == pred_label else 'red'
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
            axes[i, j].set_title(title, fontsize=10, color=color, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(RESULTS_DIR, 'sample_predictions.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Sample predictions saved to: {output_path}")
        
        plt.show()
        print("📊 Close the window to continue...")
        
        break

def main():
    """Main testing pipeline"""
    
    print("=" * 50)
    print("MODEL TESTING PIPELINE")
    print("=" * 50)
    
    # Load model
    model, class_names = load_model_and_classes()
    
    # Load test data
    test_ds = load_test_data(class_names)
    
    # Evaluate
    results = evaluate_model(model, test_ds, class_names)
    
    # Confusion matrix
    create_confusion_matrix(model, test_ds, class_names)
    
    # Visualize predictions
    visualize_predictions(model, test_ds, class_names)
    
    print("\n" + "=" * 50)
    print("✅ TESTING COMPLETE!")
    print("=" * 50)
    print(f"\n🎯 Final Test Accuracy: {results[1]*100:.2f}%")
    print("\nYour model is ready for deployment! 🚀")

if __name__ == "__main__":
    main()