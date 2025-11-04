# movie_rating_prediction.py
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- paths
DATA_IN = os.path.join("data", "IMDb Movies India.csv")
PROCESSED_OUT = os.path.join("data", "processed.csv")
META_OUT = os.path.join("data", "features_meta.pkl")
MODEL_OUT = os.path.join("models", "random_forest.pkl")
METRICS_OUT = os.path.join("models", "metrics.pkl")
IMG_DIR = "images"
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

if not os.path.exists(DATA_IN):
    print("ERROR: dataset missing at", DATA_IN)
    sys.exit(1)

df = pd.read_csv(DATA_IN, encoding="latin1")
df.columns = [c.strip() for c in df.columns.tolist()]

# normalize columns
rename_map = {}
if 'Name' in df.columns: rename_map['Name'] = 'Title'
if 'Actor 1' in df.columns: rename_map['Actor 1'] = 'Actor1'
if 'Actor 2' in df.columns: rename_map['Actor 2'] = 'Actor2'
if 'Actor 3' in df.columns: rename_map['Actor 3'] = 'Actor3'
df = df.rename(columns=rename_map)

# keep necessary columns, fill defaults
for c in ['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3', 'Duration', 'Votes', 'Rating', 'Title']:
    if c not in df.columns:
        df[c] = np.nan

df = df[df['Rating'].notna()].copy()

# parse duration to minutes
def parse_duration(x):
    try:
        return float(x)
    except:
        if isinstance(x, str):
            s = x.lower().replace('mins','m').replace('min','m').replace('hrs','h').replace('hr','h')
            parts = s.replace('-', ' ').split()
            mins = 0
            for p in parts:
                if 'h' in p:
                    digits = ''.join(ch for ch in p if ch.isdigit())
                    if digits: mins += int(digits) * 60
                elif 'm' in p:
                    digits = ''.join(ch for ch in p if ch.isdigit())
                    if digits: mins += int(digits)
            return float(mins) if mins>0 else 0.0
    return 0.0

df['Duration'] = df['Duration'].apply(parse_duration)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0).astype(int)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

# normalize names
for c in ['Director','Actor1','Actor2','Actor3','Genre','Title']:
    if c in df.columns:
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()

# multi-genre handling
def split_genres(g):
    if pd.isna(g): return []
    if isinstance(g, str):
        parts = [p.strip() for p in g.split(',') if p.strip()]
        return parts if len(parts)>0 else ['Unknown']
    return ['Unknown']

df['genre_list'] = df['Genre'].apply(split_genres)

mlb = MultiLabelBinarizer(sparse_output=False)
genre_mtx = mlb.fit_transform(df['genre_list'])
genre_cols = [f"genre_{g}" for g in mlb.classes_]
genre_df = pd.DataFrame(genre_mtx, columns=genre_cols, index=df.index)
df = pd.concat([df, genre_df], axis=1)

# label encode director & actors and create mapping dicts
def make_map(series):
    vals = series.fillna('Unknown').astype(str)
    le = LabelEncoder()
    le.fit(vals)
    classes = le.classes_.tolist()
    mapping = {cls: int(i) for i, cls in enumerate(classes)}
    return mapping

director_map = make_map(df['Director'])
actor1_map = make_map(df['Actor1'])
actor2_map = make_map(df['Actor2'])
actor3_map = make_map(df['Actor3'])

# mapping function with fallback to 'Unknown' if not present
def map_with_fallback(name, mapping):
    if name in mapping:
        return mapping[name]
    if 'Unknown' in mapping:
        return mapping['Unknown']
    # fallback to 0
    return 0

df['Director_enc'] = df['Director'].apply(lambda x: map_with_fallback(x, director_map))
df['Actor1_enc'] = df['Actor1'].apply(lambda x: map_with_fallback(x, actor1_map))
df['Actor2_enc'] = df['Actor2'].apply(lambda x: map_with_fallback(x, actor2_map))
df['Actor3_enc'] = df['Actor3'].apply(lambda x: map_with_fallback(x, actor3_map))

# final feature list: all genre_* columns + encodings + duration + votes
feature_cols = genre_cols + ['Director_enc','Actor1_enc','Actor2_enc','Actor3_enc','Duration','Votes']

# ensure no NaNs
X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['Rating'].astype(float)

if X.shape[0] == 0:
    print("No training samples found after preprocessing.")
    sys.exit(1)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train models
rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# predict & evaluate
pred_rf = rf.predict(X_test)
pred_ridge = ridge.predict(X_test)

def metrics_dict(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False) if 'squared' in mean_squared_error.__code__.co_varnames else np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}

metrics = {
    'random_forest': metrics_dict(y_test, pred_rf),
    'ridge': metrics_dict(y_test, pred_ridge)
}

print("Training results:")
print("RandomForest ->", metrics['random_forest'])
print("Ridge ->", metrics['ridge'])

# save model, meta, processed csv, metrics, images
joblib.dump(rf, MODEL_OUT)
joblib.dump(metrics, METRICS_OUT)

meta = {
    'mlb_classes': mlb.classes_.tolist(),
    'genre_cols': genre_cols,
    'director_map': director_map,
    'actor1_map': actor1_map,
    'actor2_map': actor2_map,
    'actor3_map': actor3_map,
    'feature_cols': feature_cols
}
joblib.dump(meta, META_OUT)

df.to_csv(PROCESSED_OUT, index=False)

# save images
plt.figure(figsize=(6,5))
plt.scatter(y_test, pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=1)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("RandomForest: Actual vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "actual_vs_predicted.png"))
plt.close()

if hasattr(rf, "feature_importances_"):
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    cols = [feature_cols[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(8, max(3, len(cols)*0.25)))
    plt.barh(cols[::-1], vals[::-1])
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"))
    plt.close()

print("Saved model:", MODEL_OUT)
print("Saved metadata:", META_OUT)
print("Saved processed data:", PROCESSED_OUT)
print("Saved metrics:", METRICS_OUT)
