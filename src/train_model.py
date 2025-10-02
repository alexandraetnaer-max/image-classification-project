"""
Train image classification model using Transfer Learning
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Paths
PROCESSED_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DIR, 'validation')
MODELS_DIR = 'models'
LOGS_DIR = 'logs'

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def create_data_generators():
    """Create data generators for training and validation"""
    
    print("\n" + "=" * 50)
    print("CREATING DATA GENERATORS")
    print("=" * 50)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Number of classes: {train_generator.num_classes}")
    print(f"\n📦 Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator

def build_model(num_classes):
    """Build model using Transfer Learning with MobileNetV2"""
    
    print("\n" + "=" * 50)
    print("BUILDING MODEL")
    print("=" * 50)
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    print(f"\n🧠 Base model: MobileNetV2")
    print(f"   Trainable: {base_model.trainable}")
    print(f"   Parameters: {base_model.count_params():,}")
    
    # Build model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📊 Model summary:")
    model.summary()
    
    return model, base_model

def train_model(model, train_generator, val_generator):
    """Train the model"""
    
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    
    print(f"\n⚙️ Training configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Image size: {IMG_SIZE}")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        keras.callbacks.CSVLogger(
            os.path.join(LOGS_DIR, f'training_log_{timestamp}.csv')
        )
    ]
    
    print(f"\n🚀 Starting training...")
    print(f"   This will take approximately 20-30 minutes...")
    
    # Train
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Training complete!")
    
    return history

def plot_training_history(history, timestamp):
    """Plot training metrics"""
    
    print("\n" + "=" * 50)
    print("PLOTTING TRAINING HISTORY")
    print("=" * 50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join('results', f'training_history_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training history plot saved to: {plot_path}")
    
    plt.show()

def save_model(model, train_generator, timestamp):
    """Save the trained model"""
    
    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f'fashion_classifier_{timestamp}.h5')
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Save class indices
    class_indices = train_generator.class_indices
    # Reverse the dictionary
    index_to_class = {v: k for k, v in class_indices.items()}
    
    metadata = {
        'class_indices': class_indices,
        'index_to_class': index_to_class,
        'num_classes': len(class_indices),
        'image_size': IMG_SIZE,
        'timestamp': timestamp
    }
    
    metadata_path = os.path.join(MODELS_DIR, f'model_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✅ Metadata saved to: {metadata_path}")
    
    return model_path

def main():
    """Main training pipeline"""
    
    print("=" * 50)
    print("MODEL TRAINING PIPELINE")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Create data generators
    train_generator, val_generator = create_data_generators()
    
    # Step 2: Build model
    model, base_model = build_model(train_generator.num_classes)
    
    # Step 3: Train model
    history = train_model(model, train_generator, val_generator)
    
    # Step 4: Plot results
    plot_training_history(history, timestamp)
    
    # Step 5: Save model
    model_path = save_model(model, train_generator, timestamp)
    
    # Final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\n📊 Final Results:")
    print(f"   Training Accuracy:   {final_train_acc:.2%}")
    print(f"   Validation Accuracy: {final_val_acc:.2%}")
    print(f"\n💾 Model saved to: {model_path}")
    print(f"\nNext step: Test the model and create API!")

if __name__ == "__main__":
    main()