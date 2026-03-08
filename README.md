# Deep Learning Challenges
## Deep Learning Challenges

This repository contains the solutions for two deep learning classification challenges developed as part of a coursework assignment. Both tasks require designing and training neural networks **from scratch**, without using pretrained models or pretrained weights.

### Challenge 1 — Multivariate Time Series Classification

The second task focuses on **classification of multivariate time series data**.

The dataset consists of sequences with **180 time steps**, where each step contains measurements from multiple input channels.

The goal is to design a neural network capable of analyzing the temporal patterns in each sequence and assigning the **entire sequence** to **one of three possible classes**.

Key characteristics of the task:

- Input: multivariate time series sequences
- Sequence length: **180 time steps**
- Multiple input channels per time step
- Goal: **sequence-level classification**
- Output: one of **three classes**

---

#### Constraints

The assignment enforces the following important rule:

- **Pretrained models are strictly forbidden.**
- All architectures must be **implemented and trained from scratch**.
- It is not allowed to rely on existing pretrained weights or pretrained backbone architectures.

---

### Challenge 2 — Histopathology Image Classification

The first task focuses on image classification in the medical domain.  
The dataset consists of images extracted from **low-magnification Whole Slide Images (WSI) of human tissue**.

The objective is to build a deep learning model that can analyze each image and assign it to **one of four possible classes**, representing different **molecular subtypes associated with a potential disease in the tissue**.

Key characteristics of the task:

- Input: tissue image patches extracted from WSI
- Goal: **multi-class image classification**
- Output: one of **four molecular subtype classes**
- Models must be **designed and trained from scratch**


