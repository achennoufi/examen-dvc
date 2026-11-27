import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

OUT = "data/processed_data"


def preprocess_dataset(df):
    # split
    df_clean = df.copy().set_index("date")
    X = df_clean.drop(columns=["silica_concentrate"])
    y = df_clean["silica_concentrate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv("data/raw_data/raw.csv")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_dataset(df)

    X_train_scaled.to_csv(f"{OUT}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{OUT}/X_test_scaled.csv", index=False)
    y_train.to_csv(f"{OUT}/y_train.csv", index=False)
    y_test.to_csv(f"{OUT}/y_test.csv", index=False)
