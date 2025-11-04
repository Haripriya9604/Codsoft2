# Codsoft2
task 2 Movie Rating Prediction
# 🎬 Movie Rating Prediction with Python (CLI-Based)

![Python](https://img.shields.io/badge/Python-3.11.6-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.2.3-green?logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.26.4-lightgrey?logo=numpy)
![matplotlib](https://img.shields.io/badge/matplotlib-3.9.2-purple?logo=matplotlib)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

> A complete Machine Learning project that predicts IMDb movie ratings based on **Genre, Director, and Actor** information.  
> It includes **end-to-end preprocessing, training, evaluation, and a real-time CLI interface** for interactive prediction.

---

## 📖 **Project Overview**

🎯 The goal is to predict the IMDb rating of a movie using structured metadata such as:
- **Genre(s)** — multi-label categorical feature (Action, Drama, Comedy, etc.)
- **Director** — label encoded
- **Actors** — encoded by unique ID
- **Duration** — numeric runtime (minutes)
- **Votes** — total number of IMDb user votes

The model learns from **historical movie data** and provides insight into what factors most influence a film’s success.

---

## 🧩 **Project Workflow**

### 🔹 1. Data Preprocessing
- Loaded dataset from [IMDb India Movies](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)
- Cleaned missing values and normalized columns
- Extracted numerical runtime from text duration values (e.g. `2h 30min → 150`)
- Converted genres into one-hot encoded features using `MultiLabelBinarizer`

### 🔹 2. Feature Engineering
| Feature Type | Example | Encoding Method |
|---------------|----------|----------------|
| Genre | Action, Comedy | MultiLabelBinarizer |
| Director | Rohit Shetty | LabelEncoder |
| Actors | Shah Rukh Khan, Ajay Devgn | LabelEncoder |
| Duration | 150 | Numeric |
| Votes | 80000 | Numeric |

### 🔹 3. Model Training
Two models were trained for comparison:
- **Random Forest Regressor (primary)**
- **Ridge Regression (baseline)**

### 🔹 4. Model Evaluation
| Metric | Random Forest | Ridge Regression |
|---------|----------------|-----------------|
| Mean Absolute Error (MAE) | 0.98 | 1.08 |
| Root Mean Squared Error (RMSE) | 1.25 | 1.34 |
| R² Score | 0.15 | 0.03 |

Random Forest achieved better performance and is saved as the final deployed model.

### 🔹 5. Model Saving
All generated artifacts are stored automatically:
models/random_forest.pkl
data/features_meta.pkl
data/processed.csv
models/metrics.pkl
images/*.png


### 🔹 6. Real-Time CLI Prediction
After training, use the **CLI interface** (`movie_predict_cli.py`) to predict IMDb ratings interactively by typing inputs such as genre, director, actor names, duration, and votes.

---

