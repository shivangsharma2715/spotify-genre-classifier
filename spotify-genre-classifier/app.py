"""
Streamlit App: Spotify Song Genre Classifier
Demonstrates 5 classification models trained to predict playlist_genre
from 13 audio/track features of a song.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Spotify Genre Classifier", page_icon="🎧", layout="wide")

FEATURE_COLS = [
    'track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]
TARGET_COL = 'playlist_genre'

MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'kNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Random Forest (Ensemble)': 'model/random_forest_ensemble.joblib',
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.joblib')
    label_encoder = joblib.load('model/label_encoder.joblib')
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, label_encoder, models


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc = np.nan
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        'Accuracy': acc, 'AUC': auc, 'Precision': prec,
        'Recall': rec, 'F1': f1, 'MCC': mcc,
    }


def main():
    st.title("🎧 Spotify Song Genre Classifier")
    st.caption(
        "Multi-class classification of song genre (edm / latin / pop / r&b / rap / rock) "
        "from 13 audio features, using 5 ML models trained on the TidyTuesday Spotify Songs dataset."
    )

    scaler, label_encoder, models = load_artifacts()

    st.sidebar.header("⚙️ Controls")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test data (CSV)", type=["csv"],
        help="Upload the provided test_data.csv, or any CSV with the same 13 feature "
             "columns plus a 'playlist_genre' column."
    )
    model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))
    compare_all = st.sidebar.checkbox("Compare all 5 models", value=False)

    if uploaded_file is None:
        st.info("👈 Upload `test_data.csv` from the sidebar to get started.")
        st.markdown(
            "Expected columns: `" + "`, `".join(FEATURE_COLS + [TARGET_COL]) + "`"
        )
        return

    df = pd.read_csv(uploaded_file)

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
        return

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    X = df[FEATURE_COLS]
    y_true = label_encoder.transform(df[TARGET_COL])
    X_scaled = scaler.transform(X)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{len(df)} rows loaded.")

    if compare_all:
        st.subheader("📊 Comparison of All 5 Models")
        rows = {}
        for name, model in models.items():
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)
            rows[name] = compute_metrics(y_true, y_pred, y_proba)
        summary = pd.DataFrame(rows).T.round(4)
        st.dataframe(summary.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        summary[['Accuracy', 'F1', 'MCC']].plot(kind='bar', ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        plt.xticks(rotation=20, ha='right')
        st.pyplot(fig)
        return

    model = models[model_name]
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    metrics = compute_metrics(y_true, y_pred, y_proba)

    st.subheader(f"📈 Evaluation Metrics — {model_name}")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.subheader("📋 Classification Report")
        report = classification_report(
            y_true, y_pred, target_names=label_encoder.classes_,
            zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("🔎 Predictions Sample")
    pred_df = df[FEATURE_COLS].copy()
    pred_df['Actual Genre'] = label_encoder.inverse_transform(y_true)
    pred_df['Predicted Genre'] = label_encoder.inverse_transform(y_pred)
    st.dataframe(pred_df.head(20), use_container_width=True)


if __name__ == '__main__':
    main()
