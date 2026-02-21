# FashionMNIST CNN Classifier with PyTorch 👕👟

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify images from the FashionMNIST dataset. The model achieves an overall test accuracy of **~91.6%** and utilizes best practices like Batch Normalization, Dropout, and Early Stopping to prevent overfitting.



## 🛠️ Tech Stack & Libraries
* **Framework:** PyTorch (`torch`, `torchvision`)
* **Data Processing:** Pandas, NumPy
* **Evaluation:** Scikit-learn (`classification_report`)

## 🧠 Model Architecture
The custom CNN architecture is designed to extract spatial features efficiently before passing them to fully connected layers for classification.



[Image of a Convolutional Neural Network architecture]


* **Layer 1:** `Conv2d` (1 input channel -> 32 output channels) + `BatchNorm2d` + ReLU + `MaxPool2d`
* **Layer 2:** `Conv2d` (32 -> 64 channels) + `BatchNorm2d` + ReLU + `MaxPool2d`
* **Flatten Layer**
* **Fully Connected 1:** Linear (3136 -> 256) + ReLU + `Dropout(0.5)`
* **Fully Connected 2:** Linear (256 -> 128) + ReLU + `Dropout(0.5)`
* **Output Layer:** Linear (128 -> 10 classes)

## 🚀 Training Features
* **Optimizer:** Adam (Learning Rate: `0.001`, Weight Decay: `1e-4` for L2 regularization).
* **Loss Function:** CrossEntropyLoss.
* **Early Stopping:** Implemented a custom early stopping mechanism with a `patience` of 3 epochs.



## 📊 Results
The model automatically halted training at **Epoch 16** due to the Early Stopping callback (preventing overfitting).

* **Training Accuracy:** ~94.2%
* **Testing Accuracy:** ~91.6%
* **F1-Scores:** Performed exceptionally well on distinct items like Trousers, Sandals, Sneakers, and Bags (F1 = 0.97 - 0.99). The lowest score was for shirts/tops (Class 6), which is standard for this dataset due to high visual similarity with other clothing categories.

## 💻 How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed (`pip install torch torchvision pandas numpy scikit-learn`).
3. Run the Jupyter Notebook cell by cell. The dataset will be downloaded automatically to a `/data` directory.