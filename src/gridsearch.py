import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

DATA = "data/processed_data"

if __name__ == "__main__":
    X_train = pd.read_csv(f"{DATA}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{DATA}/y_train.csv").squeeze()

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    joblib.dump(grid.best_params_, "models/best_params.pkl")
