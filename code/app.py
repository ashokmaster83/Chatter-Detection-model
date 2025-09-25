import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, zipfile, tempfile
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import joblib

# =======================
# 1. Helper Functions
# =======================
def clean_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            cleaned_line = line.rstrip(',\n') + '\n'
            outfile.write(cleaned_line)

def segment_file(file_path):
    cleaned_path = file_path.replace(".csv", "_cleaned.csv")
    clean_file(file_path, cleaned_path)

    df = pd.read_csv(cleaned_path)
    time_diffs = np.diff(df['Time (s)'])
    sampling_rate = int(round(1 / np.median(time_diffs)))
    samples_per_window = sampling_rate
    num_windows = len(df) // samples_per_window

    segments = []
    for i in range(num_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        segment = {
            'start_time': df['Time (s)'].iloc[start],
            'X': df['X-Axis (m/s2)'].iloc[start:end].values,
            'Y': df['Y- Axis (m/s2)'].iloc[start:end].values,
            'Z': df['Z-Axis (m/s2)'].iloc[start:end].values
        }
        segments.append(segment)

    return segments, sampling_rate

def extract_features(segment, fs):
    features = {}
    sig = segment['Z']

    # Time-domain
    features['rms'] = np.sqrt(np.mean(sig**2))
    features['std'] = np.std(sig)
    features['peak_to_peak'] = np.ptp(sig)
    features['time_skew'] = skew(sig)
    features['time_kurtosis'] = pd.Series(sig).kurtosis()

    # Frequency-domain
    freqs, psd = welch(sig, fs=fs)
    band_mask = (freqs >= 50) & (freqs <= 200)
    band_energy = np.sum(psd[band_mask])
    total_energy = np.sum(psd)

    features['band_energy_ratio'] = band_energy / total_energy if total_energy > 0 else 0
    features['dominant_freq'] = freqs[np.argmax(psd)]
    features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
    features['peak_count'] = np.sum(psd > np.mean(psd))
    features['spectral_kurtosis'] = kurtosis(psd)

    return features

# =======================
# 2. Streamlit UI
# =======================
st.title("📊 Machining Chatter Detection App")

# Load trained model automatically
MODEL_PATH = "voting_model.pkl"  # change if stored elsewhere
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Could not load model at {MODEL_PATH}: {e}")
    st.stop()

# Upload CSV ZIP
zip_file = st.file_uploader("Upload ZIP of CSV files", type="zip")

if zip_file is not None:
    # Extract to temp dir
    tempdir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(tempdir)
    st.success("✅ CSVs extracted")

    # Segment + extract features
    all_segments, segment_file_map = [], []
    sampling_rate = None

    for fname in os.listdir(tempdir):
        if fname.endswith(".csv"):
            segs, fs = segment_file(os.path.join(tempdir, fname))
            all_segments.extend(segs)
            segment_file_map.extend([fname]*len(segs))
            sampling_rate = fs  # assume consistent

    features_list = [extract_features(seg, sampling_rate) for seg in all_segments]
    features_df = pd.DataFrame(features_list)
    features_df['file'] = segment_file_map

    st.write("### Extracted Features", features_df.head())

    # Model predictions
    feature_cols = [col for col in features_df.columns if col not in ['file','prediction','cluster','cluster_mapped']]
    features_df['prediction'] = model.predict(features_df[feature_cols])

    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    features_df['cluster'] = kmeans.fit_predict(features_df[feature_cols])

    cluster_to_label = {}
    for cluster in features_df['cluster'].unique():
        majority_label = features_df[features_df['cluster']==cluster]['prediction'].mode()[0]
        cluster_to_label[cluster] = majority_label
    features_df['cluster_mapped'] = features_df['cluster'].map(cluster_to_label)

    # Metrics
    overall_agreement = np.mean(features_df['cluster_mapped']==features_df['prediction'])
    st.metric("Overall Agreement", f"{overall_agreement*100:.2f}%")

    # File-level chatter/stable result
    st.write("### File-level Results")

    file_results = []
    for fname in features_df['file'].unique():
        file_df = features_df[features_df['file'] == fname]
        majority_vote = file_df['prediction'].mode()[0]   # most common prediction
        label = "Stable" if majority_vote == 0 else "Chatter"
        agreement = np.mean(file_df['cluster_mapped']==file_df['prediction']) * 100
        file_results.append({"File": fname, "Prediction": label, "Agreement %": f"{agreement:.2f}%"})

    results_df = pd.DataFrame(file_results)
    st.dataframe(results_df, use_container_width=True)

    # Plots
    st.write("### Predictions vs Clusters")
    fig, ax = plt.subplots()
    sns.countplot(x='cluster', hue='prediction', data=features_df, palette="Set2", ax=ax)
    ax.set_title("Model Predictions vs KMeans Clusters")
    st.pyplot(fig)

    cm = confusion_matrix(features_df['cluster_mapped'], features_df['prediction'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=['Stable','Chatter'],
                yticklabels=['Stable','Chatter'], ax=ax)
    ax.set_title("Confusion Matrix: Prediction vs Cluster Majority")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.kdeplot(data=features_df, x='spectral_kurtosis',
                hue='cluster', fill=True, common_norm=False, palette="Set1", ax=ax)
    ax.set_title("KDE of Spectral Kurtosis per Cluster")
    st.pyplot(fig)
