
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from load_data import DATA_DIR


# Load the already-cleaned dataset.
df = pd.read_csv(DATA_DIR / "clean.csv").dropna(
    subset=["content", "label"]
)

# Words and phrases that may reveal writing style rather than truth.
WEB_ARTIFACTS = [
    "featured image", "getty images", "image via", "via",
    "image", "video", "watch", "read", "com", "getty",
    "screenshot", "twitter", "pic", "facebook",
]

DATE_WORDS = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "nov", "oct",
    "dec", "jan", "feb",
]

# Keep the same train/test split for every experiment.
X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)


def remove_terms(texts, terms):
    """Remove selected words and phrases from article text."""
    pattern = r"\b(?:" + "|".join(
        re.escape(term) for term in sorted(terms, key=len, reverse=True)
    ) + r")\b"

    return texts.str.replace(pattern, " ", regex=True)


experiments = {
    "Baseline": [],
    "Without web artifacts": WEB_ARTIFACTS,
    "Without web and date artifacts": WEB_ARTIFACTS + DATE_WORDS,
}

results = {}

for name, terms in experiments.items():
    train_text = X_train.copy()
    test_text = X_test.copy()

    # Apply exactly the same removal to train and test data.
    if terms:
        train_text = remove_terms(train_text, terms)
        test_text = remove_terms(test_text, terms)

    # TF-IDF learns its vocabulary from training data only.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                max_features=50000,
            ),
        ),
        ("clf", LinearSVC()),
    ])

    model.fit(train_text, y_train)
    predictions = model.predict(test_text)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)
    print(f"Accuracy: {accuracy:.4%}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["fake", "real"],
            zero_division=0,
        )
    )

baseline = results["Baseline"]

print("\n\nROBUSTNESS SUMMARY")
print("-" * 55)

for name, accuracy in results.items():
    difference = (accuracy - baseline) * 100
    print(
        f"{name:35s} "
        f"Accuracy: {accuracy:.4%} | "
        f"Change from baseline: {difference:+.2f} percentage points"
    )

print(
    "\nInterpretation: a large accuracy drop after removing certain "
    "words may indicate reliance on those textual shortcuts. "
    "It does not, by itself, prove that all remaining predictions "
    "are based on factual content."
)
