"""
Train model with proper saving (FIXED VERSION)
"""
import os
import tensorflow as tf
import keras
from keras import layers
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Disable warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths
PROCESSED_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DIR, 'validation')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Full training
LEARNING_RATE = 0.001

print("=" * 60)
print("TRAINING MODEL - FIXED VERSION")
print("=" * 60)

print("\n📋 Configuration:")
print(f"   Image size: {IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE}")

# Step 1: Load data
print("\n" + "=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

train_ds = keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
    seed=42
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n✅ Data loaded:")
print(f"   Classes: {num_classes}")
print(f"   {class_names}")

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Step 2: Build model
print("\n" + "=" * 60)
print("STEP 2: BUILDING MODEL")
print("=" * 60)

# Load base model
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

print("\n🧠 Base model: MobileNetV2 (frozen)")

# Build model
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = layers.Rescaling(1./127.5, offset=-1)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n✅ Model built with {model.count_params():,} parameters")

# Step 3: Train
print("\n" + "=" * 60)
print("STEP 3: TRAINING")
print("=" * 60)
print(f"\n🏋️ Training for {EPOCHS} epochs (approximately 15-20 minutes)...")
print("=" * 60 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# Step 4: Save model - MULTIPLE FORMATS
print("\n" + "=" * 60)
print("STEP 4: SAVING MODEL (MULTIPLE FORMATS)")
print("=" * 60)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save in Keras format (.keras - recommended)
keras_path = os.path.join(MODELS_DIR, f'fashion_model_{timestamp}.keras')
model.save(keras_path)
print(f"\n✅ Saved Keras format: {keras_path}")

# Save weights only
weights_path = os.path.join(MODELS_DIR, f'fashion_weights_{timestamp}.weights.h5')
model.save_weights(weights_path)
print(f"✅ Saved weights: {weights_path}")

# Save class names
classes_path = os.path.join(MODELS_DIR, 'class_names.json')
with open(classes_path, 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"✅ Saved classes: {classes_path}")

# Save model info
final_train_acc = float(history.history['accuracy'][-1])
final_val_acc = float(history.history['val_accuracy'][-1])

info = {
    'timestamp': timestamp,
    'num_classes': num_classes,
    'classes': class_names,
    'epochs_trained': EPOCHS,
    'final_train_accuracy': final_train_acc,
    'final_val_accuracy': final_val_acc,
    'image_size': IMG_SIZE,
    'keras_model_path': keras_path,
    'weights_path': weights_path
}

info_path = os.path.join(MODELS_DIR, 'model_info.json')
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)
print(f"✅ Saved info: {info_path}")

# Step 5: Plot results
print("\n" + "=" * 60)
print("STEP 5: PLOTTING RESULTS")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], 'o-', label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], 's-', label='Validation', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], 'o-', label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], 's-', label='Validation', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, f'training_plot_{timestamp}.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved: {plot_path}")

plt.show()
print("📊 Close the window to continue...")

# Step 6: Quick validation
print("\n" + "=" * 60)
print("STEP 6: QUICK VALIDATION")
print("=" * 60)

print("\n🧪 Testing saved model...")

# Load the saved model
loaded_model = keras.models.load_model(keras_path)

# Test on a small batch
test_batch = val_ds.take(10)
loss, acc = loaded_model.evaluate(test_batch, verbose=0)

print(f"\n✅ Loaded model test accuracy: {acc*100:.2f}%")

if acc > 0.7:
    print("✅ Model saved and loads correctly!")
else:
    print("⚠️ Warning: Accuracy seems low. May need more training.")

# Final summary
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📊 Final Results:")
print(f"   Training Accuracy:   {final_train_acc*100:.2f}%")
print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"   Loaded Model Test:   {acc*100:.2f}%")
print(f"\n💾 Model files saved in: {MODELS_DIR}/")
print(f"📈 Training plot saved in: {RESULTS_DIR}/")
print("\nNext: Test on full test set!")