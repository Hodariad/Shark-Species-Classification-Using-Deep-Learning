# Shark Species Classification 🦈  

## Introduction  
This is my first project and a significant step toward combining marine science and artificial intelligence to aid conservation efforts through technology. The goal is to classify shark species using a Convolutional Neural Network (CNN) and a well-curated dataset of shark images. This project highlights how AI can be leveraged for wildlife conservation and marine research.

## Features  
- Preprocessing and validation of shark image datasets.  
- Data augmentation techniques to improve model robustness.  
- A CNN model designed for efficient feature extraction and classification.  
- Support for 14 shark species with a high level of accuracy.  
- Deployment-ready trained model in Keras format.  

## Project Workflow  
1. **Data Preparation**  
   - Validation and removal of corrupted images.  
   - Dataset augmentation with random flips, rotations, and zooms.  

2. **Model Development**  
   - Built a CNN with TensorFlow and Keras, including convolutional and dense layers.  
   - Trained the model for 30 epochs with an 80/20 train-validation split.  

3. **Model Evaluation**  
   - Assessed performance using validation accuracy and loss metrics.  

4. **Deployment**  
   - Saved the trained model for easy integration and future use.  

## Getting Started  

### Prerequisites  
- Python 3.7+  
- TensorFlow 2.x  
- Additional libraries: `numpy`, `pandas`, `matplotlib`, `Pillow`  

Install dependencies:  
```bash
pip install numpy pandas matplotlib tensorflow
