# Machining Chatter Detection WebApp

This project provides a web application for detecting chatter in machining processes using vibration sensor data and machine learning. The app is built with Streamlit and uses a pre-trained ensemble model (`voting_model.pkl`) for predictions.

## Features
- Upload a ZIP file containing CSVs of vibration data
- Automatic data cleaning and segmentation
- Extraction of time-domain and frequency-domain features
- Chatter prediction using a trained voting classifier
- KMeans clustering for unsupervised analysis
- Visualizations: feature distributions, confusion matrix, agreement metrics

## How It Works
1. **Upload Data**: Upload a ZIP file containing your raw CSV sensor data.
2. **Data Processing**: The app cleans, segments, and extracts features from each CSV.
3. **Prediction**: The voting classifier predicts chatter/stable for each segment.
4. **Clustering**: KMeans clustering is performed for comparison.
5. **Visualization**: Results are shown as tables and plots for easy interpretation.

## Getting Started

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

### Running the App
```bash
streamlit run app.py
```

Open the provided local URL in your browser to use the app.

## File Structure
- `app.py` : Main Streamlit application
- `voting_model.pkl` : Pre-trained ensemble model
- `requirements.txt` : Python dependencies
- `README.md` : Project documentation

## Usage Notes
- Input ZIP should contain CSV files with columns: `Time (s)`, `X-Axis (m/s2)`, `Y- Axis (m/s2)`, `Z-Axis (m/s2)`
- The app processes only the Z-axis for feature extraction and prediction
- Results include both supervised (model) and unsupervised (KMeans) analysis

## Deployment
You can deploy this app locally or on Streamlit Community Cloud. For public deployment, push your code to GitHub and follow Streamlit Cloud instructions.

## License
This project is for academic and demonstration purposes.

## Contact
For questions or collaboration, contact [ashokmaster83](https://github.com/ashokmaster83).
