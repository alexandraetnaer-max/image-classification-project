"""
Diagnose model issue
"""
import os
from tensorflow import keras
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MODELS_DIR = 'models'
TRAIN_DIR = os.path.join('data', 'processed', 'train')

# Load model
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
MODEL_PATH = os.path.join(MODELS_DIR, sorted(model_files)[-1])

print("=" * 50)
print("MODEL DIAGNOSTICS")
print("=" * 50)

print(f"\n1. Model file: {MODEL_PATH}")
print(f"   File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")

print("\n2. Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)

print("\n3. Model summary:")
print(f"   Total layers: {len(model.layers)}")
print(f"   Total parameters: {model.count_params():,}")

# Check if model has learned anything
print("\n4. Checking if model has learned (weights not random)...")
first_dense = None
for layer in model.layers:
    if 'dense' in layer.name.lower():
        first_dense = layer
        break

if first_dense:
    weights = first_dense.get_weights()[0]
    print(f"   Dense layer weights mean: {np.mean(weights):.6f}")
    print(f"   Dense layer weights std: {np.std(weights):.6f}")
    print(f"   (If mean~0 and std~0.001, model might not be trained)")

print("\n5. Loading TRAINING data to check accuracy...")
train_ds = keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

# Recompile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n6. Testing on TRAINING data (should be ~90% if model trained)...")
# Take only small subset
train_subset = train_ds.take(50)
loss, accuracy = model.evaluate(train_subset, verbose=0)

print(f"\n   Training data accuracy: {accuracy*100:.2f}%")

if accuracy < 0.5:
    print("\n❌ PROBLEM: Model accuracy is too low even on training data!")
    print("   This means the model didn't save properly or wasn't trained.")
    print("\n💡 SOLUTION: We need to retrain the model with proper saving.")
else:
    print("\n✅ Model seems trained, but might have category mismatch.")
    print("   Need to check train/test category alignment.")

print("\n" + "=" * 50)