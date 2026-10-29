
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from load_data import DATA_DIR


# 1. Load the cleaned dataset.
df = pd.read_csv(DATA_DIR / "clean.csv").dropna(
    subset=["content", "label"]
)

X = df["content"]
y = df["label"]

# 2. Use the same train/test split as the baseline.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))

# 3. Create AdaBoost using shallow decision trees.
weak_learner = DecisionTreeClassifier(
    max_depth=1,
    random_state=42,
)

adaboost = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42,
)

# 4. Convert text to TF-IDF features and train the model.
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
    ("adaboost", adaboost),
])

print("\nTraining AdaBoost model...")
model.fit(X_train, y_train)

# 5. Evaluate the model on unseen test articles.
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== ADABOOST RESULTS =====")
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
