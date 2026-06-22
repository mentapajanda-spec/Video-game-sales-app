import os
from pathlib import Path

INPUT_FEATURES = [
    "Year",
    "Genre",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
]

CATEGORICAL_FEATURES = ["Genre"]
TARGET_NAME = "Other_Sales"

DEFAULT_DATA_FILE = Path(
    os.getenv(
        "VGSales_CSV_PATH",
        str(Path(Path(__file__).resolve().anchor) / "vgsales.csv" / "vgsales.csv"),
    )
)
MODEL_FILE = Path(__file__).with_name("video_game_sales_model.joblib")


def load_dataset(csv_path: Path = None):
    """Muat dataset global video game sales dari path yang ditentukan."""
    import pandas as pd

    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {csv_path}")

    data = pd.read_csv(csv_path)
    data[TARGET_NAME] = pd.to_numeric(data[TARGET_NAME], errors="coerce")
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Year"] = data["Year"].fillna(data["Year"].median())
    data["Genre"] = data["Genre"].fillna("Unknown")

    for feature in ["NA_Sales", "EU_Sales", "JP_Sales"]:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data = data.dropna(subset=INPUT_FEATURES + [TARGET_NAME])
    return data


def get_category_options(data=None):
    """Ambil pilihan Genre untuk formulir input."""
    if data is None:
        data = load_dataset()

    return {
        "Genre": sorted(data["Genre"].fillna("Unknown").unique())
    }


def prepare_features(data, feature_columns=None):
    """Siapkan fitur untuk model Other_Sales."""
    import pandas as pd

    df = data[INPUT_FEATURES].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Year"] = df["Year"].fillna(df["Year"].median())
    df["Genre"] = df["Genre"].fillna("Unknown")

    for feature in ["NA_Sales", "EU_Sales", "JP_Sales"]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dummy_na=False)

    if feature_columns is not None:
        for missing_col in set(feature_columns) - set(df.columns):
            df[missing_col] = 0
        df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def train_model(data):
    """Latih model regresi linear untuk memprediksi Other_Sales."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X = prepare_features(data)
    y = data[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, list(X.columns), X_train, X_test, y_train, y_test


def evaluate_model(model, X, y):
    """Evaluasi model menggunakan metrik regresi standar."""
    from sklearn import metrics
    import numpy as np

    y_pred = model.predict(X)
    return {
        "R2": float(metrics.r2_score(y, y_pred)),
        "MAE": float(metrics.mean_absolute_error(y, y_pred)),
        "MSE": float(metrics.mean_squared_error(y, y_pred)),
        "RMSE": float(np.sqrt(metrics.mean_squared_error(y, y_pred))),
    }


def save_model(model, feature_columns, path: Path = MODEL_FILE):
    """Simpan model dan metadata kolom fitur."""
    import joblib

    payload = {
        "model": model,
        "feature_columns": feature_columns,
    }
    joblib.dump(payload, path)
    return path


def load_model(path: Path = MODEL_FILE):
    """Muat model beserta metadata kolom fitur."""
    import joblib

    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload or "feature_columns" not in payload:
        raise ValueError("File model tidak berisi metadata yang diharapkan.")

    return payload


def predict_other_sales(model_payload, sample: dict):
    """Prediksi penjualan Other berdasarkan input NA/EU/JP."""
    import pandas as pd

    model = model_payload["model"]
    feature_columns = model_payload["feature_columns"]
    sample_df = pd.DataFrame([sample], columns=INPUT_FEATURES)
    X = prepare_features(sample_df, feature_columns=feature_columns)
    return float(model.predict(X)[0])


def example_input():
    """Contoh input untuk prediksi Other_Sales."""
    return {
        "Year": 2013,
        "Genre": "Action",
        "NA_Sales": 7.01,
        "EU_Sales": 9.27,
        "JP_Sales": 0.97,
    }


def main():
    data = load_dataset()
    model, feature_columns, X_train, X_test, y_train, y_test = train_model(data)

    print("Model video game sales berhasil dilatih")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Koefisien: setidaknya {len(feature_columns)} fitur")

    print("\nEvaluasi data latih:")
    print(evaluate_model(model, X_train, y_train))
    print("\nEvaluasi data uji:")
    print(evaluate_model(model, X_test, y_test))

    path = save_model(model, feature_columns)
    print(f"\nModel tersimpan di: {path}")

    contoh = example_input()
    prediksi_other = predict_other_sales({"model": model, "feature_columns": feature_columns}, contoh)
    prediksi_global = contoh["NA_Sales"] + contoh["EU_Sales"] + contoh["JP_Sales"] + prediksi_other
    print("\nContoh prediksi:")
    print(f"  Input: {contoh}")
    print(f"  Prediksi Other Sales: {prediksi_other:.4f} juta unit")
    print(f"  Prediksi Global Sales: {prediksi_global:.4f} juta unit")


if __name__ == "__main__":
    main()
