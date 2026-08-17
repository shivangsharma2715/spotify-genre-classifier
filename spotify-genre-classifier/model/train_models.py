"""
train_models.py
----------------
Trains 5 classification models on the Spotify Songs dataset to predict
`playlist_genre` (6-class classification: edm, latin, pop, r&b, rap, rock)
from 13 audio/track features.

Models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors Classifier
    4. Gaussian Naive Bayes
    5. Random Forest Classifier (Ensemble)

Outputs:
    - model/*.joblib          -> trained model objects
    - model/scaler.joblib      -> fitted StandardScaler
    - model/label_encoder.joblib -> fitted LabelEncoder for the target
    - model/feature_columns.joblib -> list of feature column names (order matters)
    - test_data.csv             -> held-out 20% test split (features + true label),
                                    used for the Streamlit app and for grading
    - model/metrics_summary.csv -> comparison table of all 6 evaluation metrics
"""

import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42

FEATURE_COLS = [
    'track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]
TARGET_COL = 'playlist_genre'


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='track_id').reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def build_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12, random_state=RANDOM_STATE
        ),
        'kNN': KNeighborsClassifier(n_neighbors=15),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def evaluate(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc = np.nan
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    return {
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4),
    }


def main():
    df = load_data('/mnt/user-data/uploads/spotify.csv')

    X = df[FEATURE_COLS].copy()
    y_raw = df[TARGET_COL].copy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the raw (unscaled) test split as test_data.csv -- this is what
    # graders / the Streamlit app will upload. Scaling is applied inside the app.
    test_df = df.loc[idx_test, FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)
    test_df.to_csv('../test_data.csv', index=False)
    print(f"Saved test_data.csv with {len(test_df)} rows")

    models = build_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test, n_classes=len(le.classes_))
        results[name] = metrics

        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, f'{safe_name}.joblib', compress=3)
        print(f"Trained {name}: {metrics}")

    joblib.dump(scaler, 'scaler.joblib', compress=3)
    joblib.dump(le, 'label_encoder.joblib', compress=3)
    joblib.dump(FEATURE_COLS, 'feature_columns.joblib', compress=3)

    summary = pd.DataFrame(results).T
    summary.index.name = 'ML Model Name'
    summary.to_csv('metrics_summary.csv')
    print("\n=== Comparison Table ===")
    print(summary.to_string())

    with open('metrics_summary.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
