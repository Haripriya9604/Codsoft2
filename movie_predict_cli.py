
import os
import sys
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
META_PATH = os.path.join(PROJECT_ROOT, "data", "features_meta.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "random_forest.pkl")

def load_files():
    if not os.path.exists(META_PATH):
        print("Missing metadata:", META_PATH)
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print("Missing model:", MODEL_PATH)
        sys.exit(1)
    meta = joblib.load(META_PATH)
    model = joblib.load(MODEL_PATH)
    return meta, model

def map_name_to_enc(name, mapping):
    if name is None:
        return mapping.get('Unknown', 0) if isinstance(mapping, dict) else 0
    name = str(name).strip()
    if name in mapping:
        return int(mapping[name])
    if 'Unknown' in mapping:
        return int(mapping['Unknown'])
    # fallback: try case-insensitive match
    lower_map = {k.lower(): v for k,v in mapping.items()}
    if name.lower() in lower_map:
        return int(lower_map[name.lower()])
    # fallback to 0
    return 0

def build_feature_row(meta, genres_input, director_name, actor1_name, actor2_name, actor3_name, duration, votes):
    feature_cols = meta.get("feature_cols", [])
    genre_classes = meta.get("mlb_classes", [])
    director_map = meta.get("director_map", {})
    actor1_map = meta.get("actor1_map", {})
    actor2_map = meta.get("actor2_map", {})
    actor3_map = meta.get("actor3_map", {})

    # initialize row with zeros
    row = {c: 0 for c in feature_cols}

    # set genres: genres_input may be comma-separated string or single name
    selected_genres = []
    if genres_input:
        if isinstance(genres_input, str):
            selected_genres = [g.strip() for g in genres_input.split(",") if g.strip()]
        elif isinstance(genres_input, (list, tuple)):
            selected_genres = [str(g).strip() for g in genres_input]
    for g in selected_genres:
        col = f"genre_{g}"
        if col in row:
            row[col] = 1
        else:
            # try case-insensitive match to an available genre
            for gc in genre_classes:
                if gc.lower() == g.lower():
                    row[f"genre_{gc}"] = 1
                    break

    # map names to encodings
    if "Director_enc" in row:
        row["Director_enc"] = map_name_to_enc(director_name, director_map)
    if "Actor1_enc" in row:
        row["Actor1_enc"] = map_name_to_enc(actor1_name, actor1_map)
    if "Actor2_enc" in row:
        row["Actor2_enc"] = map_name_to_enc(actor2_name, actor2_map)
    if "Actor3_enc" in row:
        row["Actor3_enc"] = map_name_to_enc(actor3_name, actor3_map)

    # numeric features
    if "Duration" in row:
        try:
            row["Duration"] = float(duration)
        except:
            row["Duration"] = 0.0
    if "Votes" in row:
        try:
            row["Votes"] = int(votes)
        except:
            row["Votes"] = 0

    # ensure order matches feature_cols
    X = pd.DataFrame([[row[c] if c in row else 0 for c in feature_cols]], columns=feature_cols)
    return X, row

def prompt_user_and_predict(meta, model):
    print("\nEnter movie details (type 'quit' at any prompt to exit).")
    while True:
        genres_input = input("Genres (comma-separated, e.g. Action,Drama) [or press Enter to skip]: ").strip()
        if genres_input.lower() == "quit": break

        director_input = input("Director name (e.g. Rohit Shetty) [or Enter for Unknown]: ").strip()
        if director_input.lower() == "quit": break
        if director_input == "": director_input = "Unknown"

        actor1_input = input("Primary actor name (e.g. Shah Rukh Khan) [or Enter for Unknown]: ").strip()
        if actor1_input.lower() == "quit": break
        if actor1_input == "": actor1_input = "Unknown"

        actor2_input = input("Secondary actor name (optional) [or Enter for Unknown]: ").strip()
        if actor2_input.lower() == "quit": break
        if actor2_input == "": actor2_input = "Unknown"

        actor3_input = input("Tertiary actor name (optional) [or Enter for Unknown]: ").strip()
        if actor3_input.lower() == "quit": break
        if actor3_input == "": actor3_input = "Unknown"

        duration_input = input("Duration (minutes) [default 120]: ").strip()
        if duration_input.lower() == "quit": break
        duration_input = duration_input if duration_input != "" else "120"

        votes_input = input("Number of votes [default 1000]: ").strip()
        if votes_input.lower() == "quit": break
        votes_input = votes_input if votes_input != "" else "1000"

        X, row = build_feature_row(meta, genres_input, director_input, actor1_input, actor2_input, actor3_input, duration_input, votes_input)
        try:
            pred = model.predict(X)[0]
            print("\nInput features (built):")
            for k,v in row.items():
                print(f"  {k}: {v}")
            print(f"\n🎯 Predicted IMDb Rating: {pred:.2f} / 10\n")
        except Exception as e:
            print("Prediction failed:", e)

        cont = input("Predict another? (y/n) [y]: ").strip().lower()
        if cont == "n" or cont == "no":
            break

def main():
    meta, model = load_files()
    print("Loaded model and metadata.")
    # show short help
    print("Tip: Type genres as comma-separated names (e.g. Action, Drama). If a name is not found, 'Unknown' fallback is used.")
    prompt_user_and_predict(meta, model)
    print("Goodbye!")

if __name__ == "__main__":
    main()
