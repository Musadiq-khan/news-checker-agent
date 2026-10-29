
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from load_data import BASE_DIR, DATA_DIR


def main():
    # Load the same cleaned dataset used by train.py.
    df = pd.read_csv(DATA_DIR / "clean.csv").dropna()
    df["content"] = df["content"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=30),
            n_estimators=10,
            max_samples=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42,
        ),
    }

    results = []
    reports = {}

    for name, classifier in models.items():
        print(f"\nTraining and evaluating: {name}")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.9,
                    max_features=50000,
                ),
            ),
            ("clf", classifier),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # Labels: 0 = fake, 1 = real.
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        )

        matrix = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "precision_real": precision,
            "recall_real": recall,
            "f1_real": f1,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

        reports[name] = {
            "accuracy": float(accuracy),
            "precision_real": float(precision),
            "recall_real": float(recall),
            "f1_real": float(f1),
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_labels": ["fake", "real"],
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["fake", "real"],
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision (real): {precision:.4f}")
        print(f"Recall (real):    {recall:.4f}")
        print(f"F1-score (real):  {f1:.4f}")
        print("Confusion matrix [fake, real]:")
        print(matrix)

    # Save results in the project results directory.
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    csv_path = results_dir / "model_metrics.csv"
    json_path = results_dir / "evaluation_details.json"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({
            "dataset": "clean.csv",
            "test_size": 0.2,
            "random_state": 42,
            "label_mapping": {"fake": 0, "real": 1},
            "positive_class_for_summary_metrics": "real",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "models": reports,
            "note": (
                "Results are from a held-out split of this dataset. "
                "They do not establish real-world fact-checking ability."
            ),
        }, file, indent=2)

    print("\n===== RESULTS SAVED =====")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
