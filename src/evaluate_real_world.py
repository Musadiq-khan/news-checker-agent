
import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from load_data import BASE_DIR
from preprocess import clean_text


SAMPLES_DIR = BASE_DIR / "samples"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

# Filename prefix is a provisional expected label:
# fake_ = 0, real_ = 1.
LABELS = {
    "fake": 0,
    "real": 1,
}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SAMPLES_DIR.exists():
        raise FileNotFoundError(f"Samples folder not found: {SAMPLES_DIR}")

    model = joblib.load(MODEL_PATH)
    rows = []

    for file_path in sorted(SAMPLES_DIR.glob("*.txt")):
        prefix = file_path.name.split("_", 1)[0].lower()

        if prefix not in LABELS:
            print(f"Skipping file with unknown label: {file_path.name}")
            continue

        article_text = file_path.read_text(encoding="utf-8").strip()
        cleaned = clean_text(article_text)

        if not cleaned:
            print(f"Skipping empty article: {file_path.name}")
            continue

        expected_label = LABELS[prefix]
        prediction = int(model.predict([cleaned])[0])
        score = float(model.decision_function([cleaned])[0])

        rows.append({
            "filename": file_path.name,
            "expected_label": "real" if expected_label == 1 else "fake",
            "predicted_label": "real" if prediction == 1 else "fake",
            "svm_score": score,
            "correct": prediction == expected_label,
            "cleaned_word_count": len(cleaned.split()),
        })

    if not rows:
        raise ValueError(
            f"No labelled .txt articles were found in {SAMPLES_DIR}"
        )

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "real_world_evaluation.csv"
    json_path = RESULTS_DIR / "real_world_evaluation_summary.json"

    results.to_csv(csv_path, index=False)

    y_true = [
        LABELS[row["expected_label"]]
        for row in rows
    ]
    y_pred = [
        LABELS[row["predicted_label"]]
        for row in rows
    ]

    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    summary = {
        "sample_count": len(rows),
        "correct_predictions": int(results["correct"].sum()),
        "incorrect_predictions": int((~results["correct"]).sum()),
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "matrix_labels": ["fake", "real"],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["fake", "real"],
            output_dict=True,
            zero_division=0,
        ),
        "label_caveat": (
            "Expected labels were inferred from filenames, not independently "
            "verified. Satire and misinformation are different categories. "
            "Treat these results as a small exploratory test, not a reliable "
            "estimate of real-world accuracy."
        ),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n===== REAL-WORLD SAMPLE EVALUATION =====")
    print(results.to_string(index=False))
    print(f"\nCorrect: {summary['correct_predictions']} / {len(rows)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfusion matrix [rows=actual, columns=predicted]:")
    print(matrix)
    print("\nSaved files:")
    print(csv_path)
    print(json_path)
    print("\nIMPORTANT: Expected labels come from filenames and need verification.")


if __name__ == "__main__":
    main()
