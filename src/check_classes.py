"""
Check the correct order of classes
"""
import os
from tensorflow import keras

# Paths
PROCESSED_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')

# Load dataset to get class order
train_ds = keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

print("Correct class order:")
print(train_ds.class_names)

# Save to JSON
import json
with open('models/class_names.json', 'w') as f:
    json.dump(train_ds.class_names, f, indent=2)

print("\n✅ Saved correct class names to models/class_names.json")