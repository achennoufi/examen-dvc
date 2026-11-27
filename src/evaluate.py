import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = "data/processed_data"

if __name__ == "__main__":
    X_test = pd.read_csv(f"{DATA}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{DATA}/y_test.csv").squeeze()

    model = joblib.load("models/gbr_model.pkl")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    scores = {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    with open("metrics/scores.json", "w", newline="\n") as f:
        json.dump(scores, f, indent=4)
        f.write("\n")

    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("data/predictions.csv", index=False)
    print(scores)
