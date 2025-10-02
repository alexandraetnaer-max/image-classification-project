"""
Script to explore the fashion dataset
"""
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Paths
DATA_DIR = os.path.join('data', 'raw')

def explore_dataset():
    """Explore the structure of the dataset"""
    
    print("=" * 50)
    print("DATASET EXPLORATION")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist!")
        return
    
    # List all files and folders
    print(f"\nContents of {DATA_DIR}:")
    contents = os.listdir(DATA_DIR)
    for item in contents:
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    # Try to find CSV file with metadata
    csv_files = [f for f in contents if f.endswith('.csv')]
    if csv_files:
        print(f"\n✅ Found CSV file(s): {csv_files}")
        
        # Read the first CSV
        csv_path = os.path.join(DATA_DIR, csv_files[0])
        df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  - Total images: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"\n  First 5 rows:")
        print(df.head())
        
        # Check for categories
        if 'articleType' in df.columns:
            print(f"\n📦 Product Categories:")
            categories = df['articleType'].value_counts()
            print(categories.head(10))
            print(f"\n  Total categories: {len(categories)}")
    
    # Check for images folder
    images_folders = [f for f in contents if 'image' in f.lower() and os.path.isdir(os.path.join(DATA_DIR, f))]
    if images_folders:
        print(f"\n🖼️  Found image folder(s): {images_folders}")
        
        # Count images in first folder
        images_dir = os.path.join(DATA_DIR, images_folders[0])
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  - Number of images: {len(image_files)}")
    
    print("\n" + "=" * 50)
    print("Exploration complete!")
    print("=" * 50)

if __name__ == "__main__":
    explore_dataset()