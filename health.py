from flask import Flask, jsonify, render_template
import pandas as pd
import joblib

# Required so joblib can resolve the class stored inside pipeline.pkl
# (BillingFeatureEngineer is the pipeline's first step).
from preprocess import FeatureEngineering
app = Flask(__name__)

# pipeline.pkl must be trained via preprocessing.train_and_save() on the
# FULL historical training data - that's what fixes this bug. If pipeline.pkl
# was saved from an older run that only had "preprocessor"+"model" steps
# (no "feature_engineering" step), retrain it first:
#     python preprocessing.py
pipeline = joblib.load("pipeline.pkl")


@app.route("/")
def home():
    return 'hari'


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read raw new data - no engineered columns needed/expected here.
        input_df = pd.read_excel("healthcare_test.xlsx")
        # input_df = pd.read_csv("healthcare_dataset.csv")
        # NOTE: no manual feature_engineering() call. The pipeline's first
        # step (BillingFeatureEngineer, loaded from pipeline.pkl) applies
        # the SAME group means learned from the original training set to
        # every row here - regardless of how small or skewed this batch is.
        predictions = pipeline.predict(input_df)

        fe = pipeline.named_steps["feature_engineering"]
        pre = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        engineered = fe.transform(input_df)
        transformed = pre.transform(engineered)
        anomaly_scores = model.decision_function(transformed)

        output_df = input_df.copy()
        output_df["prediction"] = predictions
        output_df["anomaly_score"] = anomaly_scores
        output_df["is_flagged"] = predictions == -1

        output_file = "prediction_output.xlsx"
        output_df.to_excel(output_file, index=False)

        return jsonify({
            "message": "Prediction completed successfully.",
            "total_records": len(output_df),
            "anomalies_detected": int((predictions == -1).sum()),
            "output_file": output_file,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)