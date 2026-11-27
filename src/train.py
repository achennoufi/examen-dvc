import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DATA = "data/processed_data"

if __name__ == "__main__":
    X_train = pd.read_csv(f"{DATA}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{DATA}/y_train.csv").squeeze()

    best_params = joblib.load("models/best_params.pkl")
    model = GradientBoostingRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/gbr_model.pkl")
    print("Model saved to models/gbr_model.pkl")
