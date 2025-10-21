[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#license)
[![Repo Size](https://img.shields.io/github/repo-size/bassembomba/housing-price-prediction-model?style=for-the-badge)](https://github.com/bassembomba/housing-price-prediction-model)
[![Last Commit](https://img.shields.io/github/last-commit/bassembomba/housing-price-prediction-model?style=for-the-badge)](https://github.com/bassembomba/housing-price-prediction-model/commits/main)

# 🏡 Housing Price Prediction Model

> A comprehensive machine learning project that explores **housing data**, performs **exploratory data analysis**, applies **feature preprocessing**, and builds **predictive models** for real-estate insights — all in one Python script.

---

## 📘 Overview

This project demonstrates a **complete ML workflow** applied to a housing dataset.  
It starts with **data exploration**, proceeds through **data cleaning and encoding**, and concludes with multiple **classification and clustering models** — all written in a single, easy-to-understand Python file: `housing_prediction_model.py`.

The goal?  
To analyze the factors influencing **furnishing status** and **housing prices**, and to test how well different algorithms can predict or group similar properties.

---

## 🧠 Key Features

✨ **Exploratory Data Analysis (EDA)**  
- Histograms, density plots, and boxplots for all numerical features  
- Correlation heatmaps and pairplots  
- Insightful visualizations to understand relationships in the data  

⚙️ **Data Cleaning**  
- Handling missing values  
- Removing outliers using the IQR method  
- Dropping duplicate rows for cleaner data  

💡 **Feature Encoding & Transformation**  
- Label encoding for categorical yes/no variables  
- Conversion of furnishing status into numeric categories  

🤖 **Machine Learning Models**  
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Principal Component Analysis (PCA) for dimensionality reduction  
- K-Means and Hierarchical Clustering for unsupervised learning  

📊 **Model Evaluation**  
- Accuracy, Precision, Recall, F1-score, and Confusion Matrix  
- ROC Curve plotting for visual performance analysis  

---

🚀 Getting Started

1️⃣ Prerequisites

Make sure you have Python 3.9+ installed along with the following libraries:

pip install pandas numpy matplotlib seaborn scikit-learn joblib

2️⃣ Run the Project

Clone this repository and execute the Python script:

git clone https://github.com/bassembomba/housing-price-prediction-model.git

cd housing-price-prediction-model
python housing_prediction_model.py

All outputs (plots, metrics, and results) will appear directly in your console or pop up as visualizations.

---

🧩 Technologies Used
Category	Tools / Libraries
Programming	Python 3.9+
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	scikit-learn
Serialization	joblib

---

🎯 Objectives

Understand the correlations between numerical housing features

Predict the furnishing status of a house based on key attributes

Apply dimensionality reduction for visualization and insight

Cluster similar houses to discover natural groupings in the market

---

🌐 Visualization Highlights

The script automatically displays:

📊 Feature distributions

🔥 Correlation heatmap

🎨 PCA 2D component plot

🌀 K-Means clustering visualization

📉 ROC curve for classification models

---

💬 “Turning data into decisions — one model at a time.”

⭐ Support

If you like this project, please consider giving it a ⭐ on GitHub — it really helps!

🏁 Final Words

This project is a great example of how data analysis, feature engineering, and ML modeling can be combined seamlessly.
It’s a practical, visual, and educational journey through the end-to-end process of building a predictive model from scratch — all within one powerful Python script.
