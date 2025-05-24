import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

# Parsing argumen
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv(args.data_path)

X = df.drop("status kredit", axis=1)
y = df["status kredit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("output", exist_ok=True)
    joblib.dump(model, "output/model.pkl")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc}")
