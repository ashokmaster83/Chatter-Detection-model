# Machining Chatter Detection Project

This repository contains a complete workflow and web application for detecting chatter in machining processes using vibration sensor data and machine learning. The project includes data cleaning, feature engineering, unsupervised and supervised learning, model evaluation, and a Streamlit-based web interface for real-time predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Web Application](#web-application)
- [Installation & Usage](#installation--usage)
- [Deployment](#deployment)
- [License](#license)
- [Contact](#contact)

## Overview
Chatter is an undesirable vibration in machining that affects product quality and tool life. This project uses vibration sensor data to automatically detect chatter using a combination of signal processing and machine learning techniques.

## Features
- Data cleaning and segmentation of raw sensor CSV files
- Extraction of time-domain and frequency-domain features
- Unsupervised clustering (KMeans) to explore data structure
- Supervised classification using Random Forest, SVM, KNN, and XGBoost
- Ensemble voting classifier for robust predictions
- Model evaluation with train/test split, cross-validation, ROC/AUC analysis
- Feature importance visualization
- Streamlit web app for real-time chatter prediction and analysis

## Project Structure
```
ME623project/
├── code/
│   ├── app.py                # Streamlit web application
│   ├── ME623project.ipynb    # Main analysis notebook
│   ├── intereractiveApp.ipynb# (Optional) Interactive notebook
│   ├── voting_model.pkl      # Pre-trained ensemble model
│   ├── requirements.txt      # Python dependencies
├── images/                   # Visualizations and figures
├── projectdataset/           # Raw sensor data (CSV, XLSX)
├── labeled_segments.csv      # Labeled feature data
├── README.md                 # Project documentation
```

## Data Preparation
- Raw vibration data is stored in CSV files under `projectdataset/`.
- Data cleaning removes formatting issues (e.g., trailing commas).
- Data is segmented into 1-second windows for analysis.

## Feature Engineering
- **Time-domain features:** RMS, standard deviation, peak-to-peak, skewness, kurtosis
- **Frequency-domain features:** Band energy ratio, dominant frequency, spectral centroid, peak count, spectral kurtosis
- Features are extracted for each window and axis (X, Y, Z)
- Features are standardized for modeling

## Modeling
- **Unsupervised:** KMeans clustering to identify natural groupings (stable vs. chatter)
- **Supervised:**
  - Models: Random Forest, SVM, KNN, XGBoost
  - Train/test split and stratified sampling
  - Cross-validation and ROC/AUC analysis
  - Ensemble voting classifier for improved accuracy
- **Feature Importance:** Visualized for all models

## Web Application
- Built with Streamlit (`app.py`)
- Upload a ZIP of CSV files for analysis
- Automatic feature extraction and prediction using the trained model
- KMeans clustering for unsupervised comparison
- Visualizations: feature distributions, confusion matrix, agreement metrics

## Installation & Usage
### Prerequisites
- Python 3.8+
- All dependencies listed in `requirements.txt`
- Pre-trained model file: `voting_model.pkl` (place in the same directory as `app.py`)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ashokmaster83/Chatter-Detection-model.git
   cd Chatter-Detection-model/code
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web App
```bash
streamlit run app.py
```
Open the provided local URL in your browser to use the app.

## Deployment
You can deploy this app locally or on Streamlit Community Cloud. For public deployment, push your code to GitHub and follow Streamlit Cloud instructions.

## Usage Notes
- Input ZIP should contain CSV files with columns: `Time (s)`, `X-Axis (m/s2)`, `Y- Axis (m/s2)`, `Z-Axis (m/s2)`
- The app processes only the Z-axis for feature extraction and prediction
- Results include both supervised (model) and unsupervised (KMeans) analysis

## License
This project is for academic and demonstration purposes.

## Contact
For questions or collaboration, contact [ashokmaster83](https://github.com/ashokmaster83).
