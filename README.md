# 🧠 Alzheimer's Disease Detection System (3D ResNet & FastAPI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-purple?style=for-the-badge)

> **Hackathon Submission** > An End-to-End Deep Learning solution for the early classification of Alzheimer's Disease using volumetric MRI scans.

---

## 📌 Project Overview

Alzheimer's Disease (AD) is a progressive neurodegenerative disorder. Early diagnosis is crucial but challenging. This project leverages **3D Convolutional Neural Networks (CNNs)** to analyze the full MRI volume ($96 \times 96 \times 96$), capturing deep spatial features that 2D methods miss.

The system classifies patients into three clinical groups:
1.  **CN (Cognitive Normal):** Healthy subjects.
2.  **MCI (Mild Cognitive Impairment):** The critical early stage.
3.  **AD (Alzheimer’s Disease):** Advanced stage.

---

## 🏗️ Model Architecture: "Medium" 3D ResNet

We developed a custom, resource-efficient **3D ResNet** (Residual Network) optimized for the **AIRAWAT** supercomputing environment and local deployment.

### Key Features
* **Input:** 3D Volumetric Data (NIfTI format).
* **Backbone:** 4 Residual Blocks with increasing filter depth (16 $\rightarrow$ 128).
* **Optimization Strategy:**
    * **Loss Function:** Weighted CrossEntropy (to handle Class Imbalance).
    * **Optimizer:** AdamW with Weight Decay ($5e^{-4}$).
    * **Scheduler:** Cosine Annealing with Warm Restarts.
    * **Data Augmentation:** Random 3D Rotations, Flips, and Intensity Scaling (via MONAI).



---

## 📂 Repository Structure

```bash
Alzheimer_Project/
├── data/                   # Raw MRI Dataset (Excluded from Git)
├── models/                 # Trained Model Weights (.pth)
├── src/
│   ├── api.py              # 🚀 FastAPI Backend for Real-Time Inference
│   ├── train_task1.py      # Task 1: Binary Classification (CN vs AD)
│   ├── train_task2.py      # Task 2: Binary Classification (CN vs MCI)
│   ├── train_task3.py      # Task 3: Multi-class (CN vs MCI vs AD) - *Main Model*
├── requirements.txt        # Python Dependencies
└── README.md               # Documentation
