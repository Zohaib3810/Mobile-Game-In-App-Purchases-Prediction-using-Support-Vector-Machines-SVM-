#  Mobile Game In-App Purchases Prediction using Support Vector Machines (SVM)

This repository contains an end-to-end Machine Learning pipeline utilizing **Support Vector Machines (SVM)** to analyze and predict user spending segments in a mobile gaming application. The project addresses data preprocessing, hyperparameter optimization across multiple SVM kernels, decision boundary visualization, and handling of class imbalances via multi-class classification strategies.

---

##  Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Key Methodology](#-key-methodology)
4. [Model Performance Summary](#-model-performance-summary)
5. [Visualizations](#-visualizations)
6. [How to Run the Project](#-how-to-run-the-project)

---

##  Project Overview
Mobile game developers rely on understanding user monetization patterns to customize in-game experiences and optimize player lifetime value. This project uses player demographics, session behavior, and payment information to predict a player's spending profile (**Minnow, Dolphin, Whale**). 

We design, tune, and evaluate several Support Vector Machine (SVM) strategies to classify players effectively, including:
* **Binary Classification**: Identifying high-value targets (Whales vs. Others).
* **Multi-Class Classification**: Comparing standard One-vs-One (OvO) and One-vs-Rest (OvR) decision boundaries.

---

##  Dataset Description
The dataset (`mobile_game_inapp_purchases.csv`) tracks in-game actions and spending profiles of 3,024 players:

* **Demographics**: `Age`, `Gender`, `Country`.
* **Device & Platform**: `Device` (Android/iOS).
* **Player Engagement**: `SessionCount`, `AverageSessionLength`, `GameGenre`.
* **Monetization & Purchases**: `InAppPurchaseAmount`, `FirstPurchaseDaysAfterInstall`, `PaymentMethod`, `LastPurchaseDate`.
* **Target Feature (`SpendingSegment`)**: 
  * `Minnow`: Low spenders.
  * `Dolphin`: Medium spenders.
  * `Whale`: High-value/High spenders.

---

##  Key Methodology

### 1. Data Cleaning & Preprocessing
* **Missing Values**: Handled using row-wise drop strategy (clean sample size reduced to 2,612 rows).
* **Feature Engineering**: Dropped high-cardinality/unnecessary identification features (`UserID`, `LastPurchaseDate`).
* **Pipeline Scaling & Encoding**: Used `ColumnTransformer` to:
  * Standard scale all continuous numerical features (`StandardScaler`).
  * One-hot encode categorical features (`OneHotEncoder`).

### 2. Hyperparameter Tuning (`GridSearchCV`)
We performed cross-validation to search for optimal SVM parameters:
* **Linear Kernel**: Tuned regularization strength $C$.
* **Polynomial Kernel**: Tuned $C$, $degree$, and $gamma$.
* **RBF Kernel**: Tuned $C$ and $gamma$.

### 3. Advanced Multi-class Strategies
Compared standard multi-class models utilizing **One-vs-One (OvO)** and **One-vs-Rest (OvR)** wrappers to handle inherent class imbalance (Whales constitute ~2.2% of the active user base).

---

##  Model Performance Summary

### **Kernel Tuning Results**
| SVM Kernel | Best Parameters | CV Accuracy | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **Linear SVM** | `{'C': 100}` | **99.14%** | **99.24%** |
| **RBF SVM** | `{'C': 100, 'gamma': 'auto'}` | 98.09% | 98.09% |
| **Polynomial SVM** | `{'C': 10, 'degree': 2, 'gamma': 'scale'}` | 95.45% | 95.60% |

### **Strategy Evaluation (RBF Multi-class)**
* **One-vs-One (OvO) Accuracy**: **95%**
* **One-vs-Rest (OvR) Accuracy**: **94%**
* Both models achieved an **F1-Score of 90.00%** on the highly imbalanced 'Whale' target segment, showing strong robustness to class skewness.

---

##  Visualizations
The repository code includes steps to generate key insight visualizers:
1. **Confusion Matrix**: Visualizing error distribution across Minnow, Dolphin, and Whale classes.
2. **PCA Decision Boundaries**: Projects 59-dimensional processed features into 2D space via Principal Component Analysis (PCA) to illustrate the decision boundaries of Linear vs. RBF kernel models.

---

##  How to Run the Project

### Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution
1. Place your `mobile_game_inapp_purchases.csv` dataset in the project directory.
2. Run the main Jupyter / Google Colab Notebook sequentially to preprocess, train, tune, and evaluate the models.
