# SUPERVISED LEARNING PROJECT - DIAMOND PREDICTION

**Language:** Python  
**Environment:** Jupyter Notebook  
**README Language:** English

---

## ⭐ Project Summary
This project implements a full supervised learning workflow to **predict diamond prices** using classical machine learning algorithms.  
It follows the structure of the AAMD (Aprenentatge Automàtic i Mineria de Dades) **Assignment 1** for the GEI 2024–25 course.

The goal is to:
- load and clean the dataset (`diamonds-train.csv`, `diamonds-test.csv`),  
- detect and remove non‑physical or anomalous values,  
- perform exploratory data analysis (EDA),  
- preprocess and encode features,  
- train multiple regression models (Linear Regression, k‑NN, Neural Network),  
- evaluate them using **MAPE**,  
- and generate predictions on the test set.

---

## 🧩 Technologies & Skills Demonstrated

### **Machine Learning**
- Regression modelling  
- Linear Regression  
- k‑Nearest Neighbors (k‑NN)  
- Feedforward Neural Network (MLP)  
- Hyperparameter exploration  
- Model comparison  

### **Data Analysis & Preprocessing**
- EDA with Pandas and Matplotlib  
- Detection of invalid or “non‑physical” entries  
- Treatment of categorical variables  
- Normalisation and scaling  
- Train/validation split  
- Data cleaning and export  

### **Software Engineering**
- Modular, well‑structured notebook  
- Clear sectioning following assignment requirements  
- Reproducibility with exported CSVs  
- Use of scikit‑learn and PyTorch  

---

## 📁 Project Structure

```
 AAMD-Supervised-Learning-Diamond-Price-Prediction/
    ├── A1.ipynb                                → Notebook containing the full assignment
    ├── data/                                   → Raw dataset files
    ├── train-preprocessed.csv                  → Cleaned training data
    ├── dtest-preprocessed.csv                  → Preprocessed test data
```

### Notebook Structure

1. **Reading the data**  
2. **Preprocessing**
   - Detect non‑physical values  
   - Remove anomalies  
   - Encode categorical variables  
   - Scaling and normalisation  
3. **EDA**
   - Distributions  
   - Correlations  
   - Feature importance analysis  
4. **Model 1 — Multilinear Regression**
5. **Model 2 — k‑NN Regression**
6. **Model 3 — Neural Network**
7. **Comparison of results (MAPE)**
8. **Prediction on test set**

---

## 🔍 Project Details

### **Data Cleaning**
The notebook identifies inconsistent diamond records such as:
- impossible carat values  
- unrealistic dimensions  
- inconsistencies between price and other attributes  

These rows are removed or corrected before model training.

---

### **Modeling**
Three supervised models are trained:

#### **1. Linear Regression**
Baseline model, fast and interpretable.

#### **2. k‑Nearest Neighbors**
Tries multiple values of `k` and distance metrics to minimise MAPE.

#### **3. Neural Network (MLP)**
A configurable feed-forward network trained with PyTorch:
- Dense layers  
- ReLU activations  
- Normalised numeric input  

---

### **Evaluation**
The models are evaluated using:

- **Mean Absolute Percentage Error (MAPE)**  
- Training curves (for NN)  
- Visualisation of predictions vs real values  

The best model is selected based on validation MAPE.

---

### **Predictions**
The chosen model is applied to the provided test split, and results are saved as:
```
dtest-preprocessed.csv
```

---

## ▶️ How to Run the Project

### **1. Install dependencies**
```
pip install pandas numpy scikit-learn matplotlib torch
```

### **2. Launch Jupyter**
```
jupyter notebook
```

### **3. Open the notebook**
```
A1.ipynb
```

### **4. Ensure datasets exist**
Place all CSV files inside `AAMD-Supervised-Learning-Diamond-Price-Prediction/` or update paths inside the notebook.

### **5. Run cells in order**
- Perform cleaning  
- Train models  
- Evaluate  
- Generate final predictions  

GPU support will be detected automatically by PyTorch.

---

## ✔ Summary
This project is a complete example of a supervised regression pipeline applied to the **diamond price prediction** task.  
It integrates EDA, preprocessing, multiple ML models, evaluation with MAPE, and final prediction generation.  
It follows exactly the structure required for AAMD Assignment 1.

