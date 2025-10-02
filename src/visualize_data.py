"""
Visualize sample images from the dataset
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

# Paths
DATA_DIR = os.path.join('data', 'raw')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'styles.csv')

def visualize_samples(num_samples=9):
    """Display a grid of random sample images"""
    
    # Read metadata
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
    
    # Get top 10 categories
    top_categories = df['articleType'].value_counts().head(10).index.tolist()
    
    # Filter dataframe for top categories only
    df_filtered = df[df['articleType'].isin(top_categories)]
    
    # Sample random images
    samples = df_filtered.sample(n=min(num_samples, len(df_filtered)))
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Sample Fashion Products', fontsize=16, fontweight='bold')
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= 9:
            break
            
        # Get image path
        img_filename = f"{row['id']}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        
        # Calculate subplot position
        i = idx // 3
        j = idx % 3
        
        # Try to load and display image
        try:
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            # Add title with category and color
            title = f"{row['articleType']}\n{row['baseColour']}"
            axes[i, j].set_title(title, fontsize=10)
            
        except FileNotFoundError:
            axes[i, j].text(0.5, 0.5, 'Image\nNot Found', 
                           ha='center', va='center')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join('results', 'sample_images.png')
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    plt.show()
    print("\n📊 Displaying images... Close the window to continue.")

def plot_category_distribution():
    """Plot distribution of top categories"""
    
    # Read metadata
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
    
    # Get top 15 categories
    top_15 = df['articleType'].value_counts().head(15)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    top_15.plot(kind='bar', color='steelblue')
    plt.title('Top 15 Product Categories', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Products', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join('results', 'category_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Category distribution saved to: {output_path}")
    
    plt.show()
    print("📊 Displaying chart... Close the window to continue.")

if __name__ == "__main__":
    print("=" * 50)
    print("VISUALIZING DATASET")
    print("=" * 50)
    
    print("\n1. Creating sample images grid...")
    visualize_samples(num_samples=9)
    
    print("\n2. Creating category distribution chart...")
    plot_category_distribution()
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print("=" * 50)