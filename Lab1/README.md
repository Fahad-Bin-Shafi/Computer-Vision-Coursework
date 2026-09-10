# Skin Cancer Classification using Transfer Learning and Deep Feature Extraction

This project evaluates **deep learning transfer-learning models** and **classical machine-learning classifiers** for multi-class skin cancer image classification using the **Skin Cancer ISIC 9 Classes** dataset.

The implementation was developed and executed in **Google Colab with GPU acceleration**.

## Project Overview

The notebook compares two approaches:

1. **End-to-End Transfer Learning**
   - VGG16
   - ResNet18
   - EfficientNet-B0

2. **Deep Feature Extraction + Classical Machine Learning**
   - ResNet18 is used as a pretrained feature extractor.
   - Extracted deep features are classified using:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - K-Nearest Neighbors (KNN)
     - Linear SVM
     - RBF-SVM
     - XGBoost

The project also compares the efficiency of the transfer-learning models using:

- Number of parameters
- Model size
- FLOPs
- Inference time
- Classification accuracy

## Dataset

**Dataset:** Skin Cancer ISIC – 9 Classes  
**Source:** Kaggle  
**Kaggle Dataset ID:** `nodoubttome/skin-cancer9-classesisic`

The dataset is downloaded directly in the notebook using `kagglehub`.

```python
import kagglehub

path = kagglehub.dataset_download(
    "nodoubttome/skin-cancer9-classesisic"
)

print("Path to dataset files:", path)
```

### Dataset Split

| Split | Images |
|---|---:|
| Training | 2,239 |
| Testing | 118 |
| **Total** | **2,357** |

### Classes

The dataset contains **9 skin-lesion classes**:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

## Image Preprocessing

All images are resized to **224 × 224 pixels** and normalized using ImageNet statistics.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Data loaders use a batch size of **32**.

## Technologies and Libraries

- Python
- PyTorch
- Torchvision
- TIMM
- KaggleHub
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- THOP
- TorchInfo
- Google Colab

## Installation

Install the required dependencies:

```bash
pip install timm thop scikit-learn xgboost torchinfo kagglehub
```

The notebook automatically uses CUDA when a GPU is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

## Transfer Learning

Three pretrained CNN architectures are adapted for the **9-class classification task**:

```python
model_names = [
    "vgg16",
    "resnet18",
    "efficientnet_b0"
]
```

Each model is created with pretrained weights:

```python
model = timm.create_model(
    name,
    pretrained=True,
    num_classes=9
)
```

### Training Configuration

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | `1e-4` |
| Loss Function | CrossEntropyLoss |
| Input Resolution | 224 × 224 |

## Transfer Learning Results

### Model Performance

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|
| VGG16 | **48.31** | 48.29 | **48.61** | 41.66 | **87.32** |
| ResNet18 | 33.90 | 15.13 | 27.78 | 19.08 | 79.36 |
| EfficientNet-B0 | 47.46 | 42.87 | 47.92 | **42.56** | 82.34 |

### Observations

- **VGG16 achieved the highest accuracy** at **48.31%**.
- **VGG16 also achieved the highest AUC** at **87.32%**.
- **EfficientNet-B0 achieved the highest F1-score** at **42.56%**.
- ResNet18 produced the lowest classification performance among the three transfer-learning models in this run.

## Deep Feature Extraction

A pretrained **ResNet18** model with its classification head removed is used as the feature extractor:

```python
backbone = timm.create_model(
    "resnet18",
    pretrained=True,
    num_classes=0
)
```

Deep representations are extracted from the training and testing images and then supplied to classical machine-learning classifiers.

## Classical Machine Learning Results

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---|---:|---:|---:|---:|
| Logistic Regression | **50.00** | 47.74 | **50.00** | **44.43** |
| Decision Tree | 23.73 | 17.19 | 25.46 | 19.11 |
| Random Forest | 37.29 | 32.83 | 39.58 | 28.52 |
| K-Nearest Neighbors | 28.81 | 33.03 | 29.63 | 28.32 |
| Linear SVM | 45.76 | 45.09 | 46.53 | 42.63 |
| RBF-SVM | 49.15 | **52.82** | 49.31 | **44.43** |
| XGBoost | 43.22 | 38.15 | 44.44 | 37.53 |

### Observations

- **Logistic Regression achieved the highest accuracy: 50.00%.**
- **RBF-SVM achieved the highest precision: 52.82%.**
- Logistic Regression and RBF-SVM both obtained an **F1-score of 44.43%**.
- Decision Tree produced the lowest overall performance in this experiment.
- The best classical classifier slightly outperformed the best end-to-end transfer-learning model in accuracy.

## Computational Efficiency

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms/image) | Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| VGG16 | 134.30 | 512.32 | 30.93 | 67.49 | **48.31** |
| ResNet18 | 11.18 | 42.74 | 3.65 | **55.49** | 33.90 |
| EfficientNet-B0 | **3.98** | **15.73** | **0.77** | 62.94 | 47.46 |

### Efficiency Analysis

**VGG16**
- Highest accuracy among the transfer-learning models.
- Largest model with **134.30M parameters**.
- Requires approximately **512.32 MB**.
- Highest computational cost at **30.93 GFLOPs**.

**ResNet18**
- Fastest measured inference time at **55.49 ms/image**.
- Significantly smaller than VGG16.
- Produced lower accuracy in this experiment.

**EfficientNet-B0**
- Most compact model.
- Only **3.98M parameters**.
- Model size of approximately **15.73 MB**.
- Lowest computational requirement at **0.77 GFLOPs**.
- Maintained **47.46% accuracy**, close to VGG16 while using far fewer resources.

Therefore, **EfficientNet-B0 provides the strongest efficiency/accuracy trade-off among the evaluated transfer-learning models in this notebook**.

## Training Loss

Recorded losses during the three training epochs were:

| Model | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| VGG16 | 1.5548 | 1.0405 | 0.7376 |
| ResNet18 | 2.0449 | 1.7467 | 1.5161 |
| EfficientNet-B0 | 2.3906 | 0.7953 | **0.4714** |

All three models showed decreasing training loss across the recorded epochs.

## Evaluation Metrics

The following metrics are calculated:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score
- Multi-class ROC-AUC using One-vs-Rest
- Parameters
- Model size
- FLOPs
- Average inference time per test image

For precision, recall, and F1-score, **macro averaging** is used so that each class contributes equally to the final score.

## Project Workflow

```text
ISIC 9-Class Dataset
        |
        v
Image Preprocessing
 Resize + Normalize
        |
        +------------------------------+
        |                              |
        v                              v
 Transfer Learning              ResNet18 Backbone
        |                       Feature Extraction
        |                              |
 VGG16 / ResNet18 /                    v
 EfficientNet-B0               Classical ML Models
        |                              |
        v                              v
 Performance Evaluation         Performance Evaluation
        |                              |
        +---------------+--------------+
                        |
                        v
               Model Comparison
            Accuracy + Efficiency
```

## How to Run

### 1. Open the notebook in Google Colab

Upload:

```text
CV_BAI_033_Lab1.ipynb
```

### 2. Enable GPU

In Google Colab:

```text
Runtime → Change runtime type → GPU
```

### 3. Run all cells

The notebook will:

1. Download the Kaggle dataset.
2. Install required dependencies.
3. Load the train and test images.
4. Preprocess the images.
5. Train VGG16, ResNet18, and EfficientNet-B0.
6. Evaluate transfer-learning performance.
7. Extract deep features with ResNet18.
8. Train seven classical classifiers.
9. Calculate model efficiency metrics.
10. Display the three final comparison tables.

## Results Summary

The best results obtained in this notebook are:

| Category | Best Model | Result |
|---|---|---:|
| Transfer Learning Accuracy | VGG16 | 48.31% |
| Transfer Learning F1 | EfficientNet-B0 | 42.56% |
| Transfer Learning AUC | VGG16 | 87.32% |
| Classical ML Accuracy | Logistic Regression | **50.00%** |
| Classical ML Precision | RBF-SVM | **52.82%** |
| Classical ML F1 | Logistic Regression / RBF-SVM | **44.43%** |
| Smallest CNN | EfficientNet-B0 | 15.73 MB |
| Lowest FLOPs | EfficientNet-B0 | 0.77 G |
| Fastest CNN Inference | ResNet18 | 55.49 ms/image |

## Conclusion

The experiment demonstrates both end-to-end transfer learning and deep-feature-based machine learning for nine-class skin lesion classification.

Among the transfer-learning models, **VGG16 produced the highest classification accuracy and AUC**, while **EfficientNet-B0 achieved nearly the same accuracy with dramatically fewer parameters, a much smaller model size, and substantially lower computational cost**.

For the deep-feature approach, **Logistic Regression achieved the highest overall accuracy of 50.00%**, while **RBF-SVM achieved the highest precision**.

The recorded results therefore show that the best-performing model depends on the objective: VGG16 for transfer-learning accuracy, Logistic Regression for overall classification accuracy in the tested pipelines, and EfficientNet-B0 for computational efficiency.

## Notebook

Main implementation:

```text
CV_BAI_033_Lab1.ipynb
```

## Author

**Fahad Bin Shafi**  
BS Artificial Intelligence  
COMSATS University Islamabad, Wah Campus

---

> **Note:** The numerical results presented in this README are taken from the saved outputs of the provided Colab notebook and correspond to that specific experimental run.

