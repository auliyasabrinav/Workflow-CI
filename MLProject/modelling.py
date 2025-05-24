import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

def main(data_path):
    mlflow.sklearn.autolog()

    df = pd.read_csv(data_path)

    X = df.drop("status kredit", axis=1)
    y = df["status kredit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("output", exist_ok=True)
    model_path = "output/model.pkl"
    joblib.dump(model, model_path)

    # log model artifact ke MLflow
    mlflow.log_artifact(model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
