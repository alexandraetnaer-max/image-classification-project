@echo off
echo ========================================
echo Setting up Image Classification Project
echo ========================================

echo.
echo Step 1: Creating virtual environment...
python -m venv venv

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate

echo.
echo Step 3: Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download dataset from Kaggle
echo 2. Extract to data/raw/
echo 3. Run: python src/prepare_data.py
echo 4. Run: python src/train_model_fixed.py
echo.
pause