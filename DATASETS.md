markdown# Datasets Documentation

## Primary Dataset

### Fashion Product Images (Small)

**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small  
**License:** CC0: Public Domain  
**Size:** ~2.3 GB  
**Images:** 44,446 total images  

**Description:**
Fashion product images dataset from Myntra (Indian fashion e-commerce platform). Contains product images with metadata including categories, subcategories, colors, and product descriptions.

**Citation:**
@dataset{fashion_product_images_small,
author = {Param Aggarwal},
title = {Fashion Product Images (Small)},
year = {2019},
publisher = {Kaggle},
url = {https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small}
}

---

## Dataset Statistics

### Used in This Project

**Total Images Used:** 25,465  
**Categories:** 10  
**Split:**
- Training: 17,825 images (70%)
- Validation: 3,820 images (15%)
- Test: 3,820 images (15%)

### Category Distribution

| Category | Training | Validation | Test | Total |
|----------|----------|------------|------|-------|
| Tshirts | 4,946 | 1,060 | 1,060 | 7,066 |
| Shirts | 2,251 | 483 | 483 | 3,217 |
| Casual Shoes | 1,992 | 427 | 427 | 2,846 |
| Watches | 1,779 | 381 | 381 | 2,541 |
| Sports Shoes | 1,425 | 305 | 305 | 2,035 |
| Kurtas | 1,291 | 277 | 277 | 1,845 |
| Tops | 1,233 | 264 | 264 | 1,761 |
| Handbags | 1,231 | 264 | 264 | 1,759 |
| Heels | 926 | 198 | 198 | 1,322 |
| Sunglasses | 751 | 161 | 161 | 1,073 |

---

## Data Preparation

### Selection Criteria

1. **Top 10 Categories:** Selected based on number of available images
2. **Image Quality:** Filtered for valid image files
3. **Minimum Threshold:** Categories with at least 750 images

### Preprocessing Steps

1. **Resizing:** All images resized to 224x224 pixels
2. **Normalization:** Pixel values scaled to [0, 1] range
3. **Data Augmentation:** Applied during training
   - Random rotation (±20°)
   - Random zoom (10%)
   - Horizontal flip
   - Width/height shift (10%)

---

## Usage Rights

### Dataset License

The Fashion Product Images dataset is released under **CC0: Public Domain** license, which means:

✅ You can:
- Use commercially
- Modify
- Distribute
- Use privately

❌ No attribution required (but appreciated)

### This Project License

This project (code and documentation) is licensed under **MIT License** (see LICENSE file).

### Model Weights

The trained model weights are derived from:
1. **MobileNetV2 (ImageNet):** Apache 2.0 License
2. **Fashion Product Images:** CC0 Public Domain

Therefore, the trained model can be used under **Apache 2.0 License** terms.

---

## Ethical Considerations

### Bias and Fairness

- Dataset focuses on Indian fashion products
- May not generalize well to other fashion styles/cultures
- Recommend retraining or fine-tuning for different markets

### Privacy

- Dataset contains product images only (no personal data)
- No personally identifiable information (PII)
- Safe for commercial and research use

### Intended Use

✅ **Appropriate Uses:**
- Fashion product classification
- E-commerce automation
- Image search and retrieval
- Research and education

❌ **Inappropriate Uses:**
- Identifying individuals
- Making decisions about people
- Any use that could cause harm

---

## Acknowledgments

- **Dataset Creator:** Param Aggarwal
- **Source Platform:** Kaggle
- **Original Data:** Myntra (fashion e-commerce)
- **Model Architecture:** Google (MobileNetV2)

---

## Additional Resources

- **Kaggle Dataset:** https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- **Project Repository:** https://github.com/alexandraetnaer-max/image-classification-project
- **Documentation:** See README.md and other docs in repository

---

## Updates and Maintenance

**Last Updated:** October 2025  
**Dataset Version:** 1.0  
**Project Version:** 1.0  

For questions or issues regarding the dataset, please refer to the original Kaggle dataset page.