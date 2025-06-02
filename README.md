# MNIST Digit Classification

This project explores multiple machine learning techniques for classifying handwritten digits using the classic MNIST dataset. The goal is to evaluate the effectiveness of different modeling approaches—Random Forest classifiers, PCA-based models, K-Means clustering, and deep neural networks—while adhering to proper machine learning workflows including data preprocessing, cross-validation, and hyperparameter tuning.

## 📊 Problem Statement

**Research Question (Layman's Terms)**:  
How can we accurately classify handwritten digits using machine learning models, and which model configurations provide the best performance? This has real-world applications in postal services, banking (check processing), and document digitization.

## 🔍 Dataset

The MNIST dataset contains grayscale images of handwritten digits (0–9), each in a 28×28 pixel matrix (flattened into 784 features). The dataset includes:
- 42,000 training images (labeled)
- 28,000 test images (unlabeled, for Kaggle submission)

Dataset source: [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) (CC BY-SA 3.0 License).

## 🧰 Methods and Workflow

### 1️⃣ Data Preprocessing
- Normalized pixel values to [0,1].
- Split training data into train/validation sets.
- PCA applied for dimensionality reduction in selected models.
- Feature scaling applied when appropriate.

### 2️⃣ Models Implemented

#### Classical Machine Learning Models
- **Model 1**: Random Forest Classifier (full pixel features).
- **Model 2**: Random Forest on PCA-transformed features (95% variance retained, ~154 components).
- **Model 3**: K-Means Clustering (unsupervised, 10 clusters, label mapping).

#### Neural Network Architectures (Deep Learning)
- Experimental 2x2 design comparing:
  - 2 layers vs. 5 layers
  - 128 neurons vs. 256 neurons per layer
- ReLU activations, softmax outputs, BatchNorm, Dropout, RMSProp optimizer.
- Trained for 40 epochs with learning rate scheduling (ReduceLROnPlateau).

### 3️⃣ Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1, ROC Curves, Precision-Recall Curves.
- Confusion Matrices for all classifiers.
- Silhouette Score for K-Means clustering.
- Training time recorded for each model.
- PCA analysis: Explained variance, dimensionality reduction effectiveness.

## 📈 Key Results

| Model                               | Accuracy   | Notes                                          |
|------------------------------------|------------|-----------------------------------------------|
| Random Forest (full features)      | 96.39%     | Strong baseline                               |
| Random Forest (PCA features)       | 94.74%     | Slight drop in accuracy, improved efficiency   |
| Random Forest (corrected PCA)      | +0.03% over PCA | Fixed data leakage issue, improved generalization |
| K-Means Clustering                 | 55.02%     | Low performance, silhouette score ≈ 0.07       |
| Neural Network (5 layers, 256 neurons) | 98.24% | Best overall accuracy, trade-off with training time |

### Insights
- PCA reduces dimensions but may slightly harm performance; careful application is needed.
- K-Means struggles due to cluster overlap in high-dimensional space.
- Neural networks significantly outperform classical methods but require more training time and resources.
- Optimal architecture balances accuracy and computational efficiency.
