"""
Prepare dataset for training: filter categories, split data, organize files
"""
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# Paths
DATA_DIR = os.path.join('data', 'raw')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'styles.csv')

PROCESSED_DIR = os.path.join('data', 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DIR, 'validation')
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')

# Configuration
TOP_N_CATEGORIES = 10
MIN_IMAGES_PER_CATEGORY = 500

def select_categories():
    """Select top N categories with sufficient images"""
    
    print("\n" + "=" * 50)
    print("STEP 1: SELECTING CATEGORIES")
    print("=" * 50)
    
    # Read metadata
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
    
    print(f"\n📊 Original dataset: {len(df)} products")
    print(f"📦 Total categories: {df['articleType'].nunique()}")
    
    # Get category counts
    category_counts = df['articleType'].value_counts()
    
    # Filter categories with minimum images
    valid_categories = category_counts[category_counts >= MIN_IMAGES_PER_CATEGORY]
    
    # Select top N
    selected_categories = valid_categories.head(TOP_N_CATEGORIES).index.tolist()
    
    print(f"\n✅ Selected {len(selected_categories)} categories:")
    for i, cat in enumerate(selected_categories, 1):
        count = category_counts[cat]
        print(f"   {i}. {cat}: {count} images")
    
    # Filter dataframe
    df_filtered = df[df['articleType'].isin(selected_categories)].copy()
    
    print(f"\n📊 Filtered dataset: {len(df_filtered)} products")
    
    return df_filtered, selected_categories

def verify_images(df):
    """Check which images actually exist"""
    
    print("\n" + "=" * 50)
    print("STEP 2: VERIFYING IMAGES")
    print("=" * 50)
    
    print("\n🔍 Checking if image files exist...")
    
    existing_images = []
    missing_count = 0
    
    for idx, row in df.iterrows():
        img_filename = f"{row['id']}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        
        if os.path.exists(img_path):
            existing_images.append(idx)
        else:
            missing_count += 1
    
    df_verified = df.loc[existing_images].copy()
    
    print(f"✅ Found: {len(df_verified)} images")
    print(f"❌ Missing: {missing_count} images")
    
    return df_verified

def split_dataset(df):
    """Split dataset into train/validation/test sets"""
    
    print("\n" + "=" * 50)
    print("STEP 3: SPLITTING DATASET")
    print("=" * 50)
    
    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        stratify=df['articleType'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df['articleType'],
        random_state=42
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(train_df)} images (70%)")
    print(f"   Validation: {len(val_df)} images (15%)")
    print(f"   Test:       {len(test_df)} images (15%)")
    
    # Show distribution per category
    print(f"\n📦 Distribution per category:")
    for cat in df['articleType'].unique():
        train_count = len(train_df[train_df['articleType'] == cat])
        val_count = len(val_df[val_df['articleType'] == cat])
        test_count = len(test_df[test_df['articleType'] == cat])
        total = train_count + val_count + test_count
        print(f"   {cat}: {total} (train:{train_count}, val:{val_count}, test:{test_count})")
    
    return train_df, val_df, test_df

def organize_files(train_df, val_df, test_df):
    """Organize images into train/val/test folders by category"""
    
    print("\n" + "=" * 50)
    print("STEP 4: ORGANIZING FILES")
    print("=" * 50)
    
    print("\n📁 Creating folder structure...")
    
    # Create directories
    for base_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for category in train_df['articleType'].unique():
            category_dir = os.path.join(base_dir, category)
            os.makedirs(category_dir, exist_ok=True)
    
    # Copy files
    datasets = [
        (train_df, TRAIN_DIR, "Training"),
        (val_df, VAL_DIR, "Validation"),
        (test_df, TEST_DIR, "Test")
    ]
    
    for df, base_dir, name in datasets:
        print(f"\n📋 Copying {name} images...")
        copied = 0
        
        for idx, row in df.iterrows():
            src_path = os.path.join(IMAGES_DIR, f"{row['id']}.jpg")
            dst_dir = os.path.join(base_dir, row['articleType'])
            dst_path = os.path.join(dst_dir, f"{row['id']}.jpg")
            
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
                
                if copied % 500 == 0:
                    print(f"   Copied {copied}/{len(df)} images...")
                    
            except Exception as e:
                print(f"   Error copying {row['id']}.jpg: {e}")
        
        print(f"   ✅ Copied {copied} images to {name}")
    
    print("\n✅ File organization complete!")

def save_metadata(train_df, val_df, test_df, categories):
    """Save processed metadata"""
    
    print("\n" + "=" * 50)
    print("STEP 5: SAVING METADATA")
    print("=" * 50)
    
    # Save split datasets
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'test_metadata.csv'), index=False)
    
    # Save category list
    with open(os.path.join(PROCESSED_DIR, 'categories.txt'), 'w') as f:
        for cat in categories:
            f.write(f"{cat}\n")
    
    print("\n✅ Metadata saved:")
    print(f"   - train_metadata.csv")
    print(f"   - val_metadata.csv")
    print(f"   - test_metadata.csv")
    print(f"   - categories.txt")

def main():
    """Main pipeline"""
    
    print("=" * 50)
    print("DATA PREPARATION PIPELINE")
    print("=" * 50)
    
    # Step 1: Select categories
    df_filtered, categories = select_categories()
    
    # Step 2: Verify images exist
    df_verified = verify_images(df_filtered)
    
    # Step 3: Split dataset
    train_df, val_df, test_df = split_dataset(df_verified)
    
    # Step 4: Organize files
    organize_files(train_df, val_df, test_df)
    
    # Step 5: Save metadata
    save_metadata(train_df, val_df, test_df, categories)
    
    print("\n" + "=" * 50)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 50)
    print("\nNext step: Train the model!")

if __name__ == "__main__":
    main()