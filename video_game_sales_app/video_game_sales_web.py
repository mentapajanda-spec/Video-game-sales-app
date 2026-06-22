from pathlib import Path

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split

import video_game_sales_service as svc

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

MODEL_PATH = Path(__file__).with_name("video_game_sales_model.joblib")


def get_or_train_model():
    try:
        return svc.load_model(MODEL_PATH)
    except FileNotFoundError:
        data = svc.load_dataset()
        model, feature_columns, _, _, _, _ = svc.train_model(data)
        svc.save_model(model, feature_columns, MODEL_PATH)
        return svc.load_model(MODEL_PATH)


def build_evaluation_metrics(model_payload):
    data = svc.load_dataset().dropna(subset=svc.INPUT_FEATURES + [svc.TARGET_NAME])
    X = svc.prepare_features(data, feature_columns=model_payload["feature_columns"])
    y = data[svc.TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )
    return {
        "train": svc.evaluate_model(model_payload["model"], X_train, y_train),
        "test": svc.evaluate_model(model_payload["model"], X_test, y_test),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    model_payload = get_or_train_model()
    metrics = build_evaluation_metrics(model_payload)
    prediction = None
    sample = None
    error_message = None

    category_options = svc.get_category_options()

    if request.method == "POST":
        try:
            sample = {}
            for feature in svc.INPUT_FEATURES:
                if feature in category_options:
                    sample[feature] = request.form.get(feature, "").strip()
                else:
                    sample[feature] = float(request.form.get(feature, "0"))
            predicted_other = svc.predict_other_sales(model_payload, sample)
            prediction = {
                "other_sales": predicted_other,
                "global_sales": sample["NA_Sales"] + sample["EU_Sales"] + sample["JP_Sales"] + predicted_other,
            }
        except ValueError:
            error_message = "Semua nilai numerik harus berupa angka."
        except Exception as exc:
            error_message = f"Terjadi kesalahan: {exc}"

    return render_template(
        "index_vg.html",
        feature_names=svc.INPUT_FEATURES,
        feature_metadata={
            "Year": {
                "label": "Tahun Rilis",
                "description": "Tahun rilis video game.",
                "example": 2013,
            },
            "Genre": {
                "label": "Genre",
                "description": "Jenis game.",
                "example": "Action",
            },
            "NA_Sales": {
                "label": "Penjualan NA (juta unit)",
                "description": "Penjualan di Amerika Utara dalam juta unit.",
                "example": 7.01,
            },
            "EU_Sales": {
                "label": "Penjualan EU (juta unit)",
                "description": "Penjualan di Eropa dalam juta unit.",
                "example": 9.27,
            },
            "JP_Sales": {
                "label": "Penjualan JP (juta unit)",
                "description": "Penjualan di Jepang dalam juta unit.",
                "example": 0.97,
            },
        },
        category_options=category_options,
        metrics=metrics,
        prediction=prediction,
        sample=sample,
        error_message=error_message,
        model_path=MODEL_PATH.name,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)
