"""
Simple model test
"""
import os
import numpy as np
from tensorflow import keras
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths
TEST_DIR = os.path.join('data', 'processed', 'test')
MODELS_DIR = 'models'

# Find model
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
MODEL_PATH = os.path.join(MODELS_DIR, sorted(model_files)[-1])

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

# Recompile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Loading test data...")
test_ds = keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

print("\nClass names from dataset:")
print(test_ds.class_names)

print("\nEvaluating model...")
loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\n" + "=" * 50)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print("=" * 50)