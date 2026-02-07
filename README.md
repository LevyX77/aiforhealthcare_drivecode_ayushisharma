# 🧠 Alzheimer's Disease Detection using 3D ResNet & FastAPI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-purple?style=for-the-badge)

> **Hackathon Project** > An End-to-End Deep Learning solution for the early classification of Alzheimer's Disease from 3D MRI Scans, featuring a custom **3D ResNet architecture** and a real-time **REST API** for inference.

---

## 📌 Project Overview

Alzheimer's Disease (AD) is often diagnosed too late. This project leverages **3D Convolutional Neural Networks (CNNs)** to analyze the entire volume of the brain (MRI) rather than 2D slices.

We classify subjects into 3 clinical stages:
1.  **CN (Cognitive Normal):** Healthy control subjects.
2.  **MCI (Mild Cognitive Impairment):** Early stage (Critical for early intervention).
3.  **AD (Alzheimer’s Disease):** Advanced stage.

---

## 🏗️ Architecture: The "Medium" 3D ResNet

Standard ResNet models (like ResNet-50) are too heavy for many medical datasets and limited GPU resources. We designed a **Custom Lightweight 3D ResNet**.

### 🔧 Key Technical Features
* **Input:** Volumetric MRI Data ($96 \times 96 \times 96$ voxels).
* **Backbone:** 4 Residual Blocks with 3D Convolutions (`Conv3d`).
* **Filters:** Progressive depth (16 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128 filters).
* **Optimization:**
    * **Loss:** Weighted CrossEntropy (Handling Class Imbalance).
    * **Optimizer:** AdamW + Cosine Annealing Warm Restarts.
    * **Augmentation:** MONAI (Random Flip, Rotation, Gaussian Noise).



---

## 📂 Project Structure

```bash
Alzheimer_Project/
├── data/                   # Raw MRI Data (Not on Git)
├── models/                 # Trained .pth models (Not on Git)
├── src/
│   ├── api.py              # FastAPI Backend for Demo
│   ├── train_task1.py      # Binary Classification (CN vs AD)
│   ├── train_task2.py      # Binary Classification (CN vs MCI)
│   └── train_task3.py      # Multi-class Model (CN vs MCI vs AD) - *Main*
├── requirements.txt        # Dependencies
└── README.md               # Documentation
