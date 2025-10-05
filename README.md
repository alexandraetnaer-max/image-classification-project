# Fashion Product Image Classification
## 🐳 Quick Start with Docker (Recommended)

If you have Docker installed, this is the easiest way:
```bash
# Clone repository
git clone https://github.com/alexandraetnaer-max/image-classification-project.git
cd image-classification-project

# Build Docker image
docker-compose build

# Start container
docker-compose run ml-training bash

# Inside container:
python src/prepare_data.py
python src/train_model_fixed.py
Machine Learning project for classifying fashion products using Transfer Learning (MobileNetV2).
See DOCKER_INSTRUCTIONS.md for detailed instructions.
4. **Результат должен выглядеть так:**
```markdown
# Fashion Product Image Classification

## 🐳 Quick Start with Docker (Recommended)

If you have Docker installed, this is the easiest way:
```bash
# Clone repository
git clone https://github.com/alexandraetnaer-max/image-classification-project.git
cd image-classification-project

# Build Docker image
docker-compose build

# Start container
docker-compose run ml-training bash

# Inside container:
python src/prepare_data.py
python src/train_model_fixed.py
See DOCKER_INSTRUCTIONS.md for detailed instructions.


## 📋 Project Overview

- **Dataset**: Fashion Product Images from Kaggle
- **Categories**: 10 classes (Tshirts, Shirts, Shoes, Watches, etc.)
- **Model**: MobileNetV2 with Transfer Learning
- **Target Accuracy**: 85-90%

## 🚀 Setup Instructions

### 1. Clone Repository
git clone https://github.com/alexandraetnaer-max/image-classification-project.git
cd image-classification-project
### 2. Create Virtual Environment
python -m venv venv
Activate it:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 3. Install Dependencies
pip install -r requirements.txt
### 4. Download Dataset

1. Register on Kaggle: https://www.kaggle.com/
2. Download dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
3. Extract to: `data/raw/`

Your structure should be:
data/raw/
├── images/
├── myntradataset/
└── styles.csv
### 5. Prepare Data
python src/prepare_data.py
This will select top 10 categories and split into train/val/test.

### 6. Train Model
python src/train_model_fixed.py
Training takes 30-40 minutes. Requires ~8GB RAM.

If memory error: edit `train_model_fixed.py` and change `BATCH_SIZE = 16` to `BATCH_SIZE = 8`

### 7. Test Model
python src/simple_test.py
## 📁 Project Structure
image-classification-project/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Prepared data
├── models/               # Trained models
├── src/                  # Source code
├── results/              # Visualizations
├── requirements.txt
└── README.md
## 🔧 Troubleshooting

**Memory Error**: Reduce BATCH_SIZE to 8 or 4

**Import errors**: Reinstall dependencies: `pip install -r requirements.txt`

## 📊 Expected Results

Test accuracy: 85-90%

## 👤 Author

Alexandra Etnaer-Max
