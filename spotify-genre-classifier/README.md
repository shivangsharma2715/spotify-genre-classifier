# 🎧 Spotify Song Genre Classifier

## a. Problem Statement

Music streaming platforms like Spotify tag songs with a playlist genre, but this
tag is assigned manually/editorially and is not always consistent. This project
builds a **multi-class classification system** that predicts a song's genre
(`edm`, `latin`, `pop`, `r&b`, `rap`, or `rock`) purely from its **audio
features** (danceability, energy, tempo, valence, etc.) — the kind of numeric
signal Spotify's own API exposes for every track. Automating genre tagging from
audio features has real applications in playlist curation, recommendation
systems, and catalog organization.

## b. Dataset Description

- **Source**: [TidyTuesday Spotify Songs dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21)
  (collected via the `spotifyr` package by Kaylin Pavlik)
- **Size used**: 28,356 unique tracks (after removing duplicate `track_id`s from
  the original 32,833 rows), well above the assignment's 500-instance minimum
- **Target variable**: `playlist_genre` — 6 classes (edm, latin, pop, r&b, rap, rock),
  reasonably balanced (~4,950–6,050 tracks per class)
- **Features used (13, ≥12 required)**:

  | Feature | Description |
  |---|---|
  | `track_popularity` | Spotify popularity score (0–100) |
  | `danceability` | How suitable the track is for dancing |
  | `energy` | Perceptual measure of intensity/activity |
  | `key` | Estimated musical key |
  | `loudness` | Overall loudness in dB |
  | `mode` | Modality (major=1 / minor=0) |
  | `speechiness` | Presence of spoken words |
  | `acousticness` | Confidence measure of whether the track is acoustic |
  | `instrumentalness` | Predicts whether a track has no vocals |
  | `liveness` | Detects presence of a live audience |
  | `valence` | Musical positiveness conveyed by the track |
  | `tempo` | Estimated tempo in BPM |
  | `duration_ms` | Track duration in milliseconds |

- **Preprocessing**: dropped duplicate tracks and rows with missing values in
  the modeling columns; features standardized with `StandardScaler` (fit on the
  training split only); target label-encoded with `LabelEncoder`; stratified
  80/20 train/test split (`random_state=42`).

## c. GitHub Repository Link

> _[ADD YOUR GITHUB REPO LINK HERE AFTER UPLOADING]_

## d. Models Used

All 5 models below were trained on the **same** preprocessed dataset and
evaluated on the same held-out 20% test split (5,672 tracks).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4709 | 0.7913 | 0.4615 | 0.4670 | 0.4619 | 0.3639 |
| Decision Tree | 0.4626 | 0.7348 | 0.4640 | 0.4593 | 0.4613 | 0.3538 |
| kNN | 0.4903 | 0.7986 | 0.4878 | 0.4867 | 0.4853 | 0.3876 |
| Naive Bayes | 0.4503 | 0.7755 | 0.4545 | 0.4476 | 0.4467 | 0.3406 |
| Random Forest (Ensemble) | 0.5631 | 0.8503 | 0.5563 | 0.5580 | 0.5525 | 0.4751 |

_(Precision, Recall, F1 are macro-averaged across the 6 classes; AUC is macro
one-vs-rest.)_

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid baseline (~47% accuracy, AUC 0.79). As a linear model it struggles where genre boundaries are non-linear in feature space (e.g., separating `pop` from `r&b`, which overlap heavily in danceability/energy), but it trains fast and is easy to interpret via coefficients. |
| Decision Tree | Weakest AUC (0.73) among all models despite similar accuracy to Logistic Regression — a single tree overfits to specific feature thresholds and doesn't generalize the class boundaries as smoothly, especially with only depth-12 pruning. |
| kNN | Second-best performer (49% accuracy, AUC 0.80). Because genre-correlated audio features form loosely clustered neighborhoods (e.g., high-tempo/high-energy tracks cluster near EDM), a distance-based approach captures this local structure better than linear or single-tree models. |
| Naive Bayes | Lowest accuracy (45%) — its core assumption that features are conditionally independent given the class is clearly violated here (e.g., `energy` and `loudness` are strongly correlated), which hurts its decision boundary. |
| Random Forest (Ensemble) | Clear winner across every metric (56.3% accuracy, AUC 0.85, MCC 0.48). Averaging 150 decision trees over bootstrapped samples and feature subsets reduces the overfitting/variance problem of a single tree while still capturing non-linear interactions between features — exactly the combination this dataset needs. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — highest score on all 6 metrics, most notably AUC (0.85) and MCC (0.48), the two metrics most robust to the dataset's class balance. |

> Note: genre classification from audio features alone is a genuinely hard
> problem — even Kaylin Pavlik's original blog post (linked in the dataset
> description) reports similar ~50–60% accuracy ranges, since genres like
> `pop`, `r&b`, and `latin` overlap significantly in audio characteristics.
> These are not classification bugs; they reflect real ambiguity in the data.

## Project Structure

```
spotify-genre-classifier/
│-- app.py                     # Streamlit web app
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out 20% test split (5,672 rows)
└-- model/
    │-- train_models.py        # full training + evaluation pipeline
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- random_forest_ensemble.joblib
    │-- scaler.joblib
    │-- label_encoder.joblib
    │-- feature_columns.joblib
    │-- metrics_summary.csv
    └-- metrics_summary.json
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv` (or any CSV with the same 13 feature columns + a
`playlist_genre` column) in the sidebar, pick a model, and view metrics, the
confusion matrix, and classification report.

## Live App

> _[ADD YOUR STREAMLIT COMMUNITY CLOUD LINK HERE AFTER DEPLOYMENT]_

## Retraining

To reproduce training from scratch:

```bash
cd model
python train_models.py
```

This regenerates all `.joblib` model files, `test_data.csv`, and
`metrics_summary.csv`/`.json`.
