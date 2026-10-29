
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from load_data import DATA_DIR


# 1. Load the cleaned dataset.
df = pd.read_csv(DATA_DIR / "clean.csv").dropna(
    subset=["content", "label"]
)

X = df["content"]
y = df["label"]

# 2. Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))

# 3. Create the Bagging classifier.
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=30,
        random_state=42,
    ),
    n_estimators=10,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)

# 4. Convert article text to TF-IDF features,
# then train Bagging on those numerical features.
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
    ("bagging", bagging),
])

print("\nTraining Bagging model...")
model.fit(X_train, y_train)

# 5. Evaluate on the held-out test set.
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== BAGGING RESULTS =====")
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
