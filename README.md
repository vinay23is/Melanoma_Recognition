# Melanoma Recognition

A CNN image classifier that sorts dermatology images into 9 skin lesion types (including melanoma), built as a deep-learning practice project around handling class imbalance and overfitting.

## What this covers
This project practices the full applied-CNN workflow on a small, imbalanced medical imaging dataset: building a baseline model, diagnosing overfitting/underfitting from training curves, using data augmentation to fix it, and using synthetic oversampling to fix class imbalance. It's a good example of iterating on a model based on what the metrics are actually telling you, rather than just running one training pass.

## Tech Stack
- Python, TensorFlow/Keras
- NumPy, Pandas, Matplotlib
- Augmentor (synthetic image oversampling)
- Google Colab (GPU training)

## Approach
- **Dataset**: 2,239 training / 118 test images from the ISIC (International Skin Imaging Collaboration) skin cancer dataset, split across 9 classes (melanoma, nevus, basal cell carcinoma, actinic keratosis, dermatofibroma, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, vascular lesion). Images resized to 180x180, batch size 32.
- **Baseline CNN** (3 conv blocks, 20 epochs): reached 84% training accuracy but only 52% validation accuracy — a clear overfitting signature (~32 point gap, rising validation loss).
- **Data augmentation** (random flips/rotation/zoom added, 20 epochs): closed the overfitting gap (61% train vs. 54% val accuracy) but the model was now underfitting — both numbers too low, meaning more signal was needed rather than less variance.
- **Class imbalance found**: raw training set ranged from 462 images (pigmented benign keratosis) down to 77 (seborrheic keratosis). Used the `Augmentor` library to generate synthetic samples (rotation, flips) and roughly balanced every class to 600-900+ images, growing the training set from 2,239 to 6,739 images.
- **Final model** (augmentation + rebalanced data, 50 epochs): reached 82% training accuracy and 79% validation accuracy — no more overfitting or underfitting. Accuracy on the held-out test set was 44.9%, showing the real generalization gap between validation and a true unseen test set that's worth being upfront about in an interview.

## Running Locally
Open `Melanoma recognition.ipynb` in Jupyter or Google Colab. The notebook downloads the ISIC dataset automatically from Google Drive via a shell cell. Requires `tensorflow`, `numpy`, `pandas`, `matplotlib`, `Pillow`, and `Augmentor` (`pip install Augmentor`). Runs end-to-end on Colab's free GPU tier; expect the augmentation/rebalancing step to take a few minutes since it generates 500 synthetic images per class.
