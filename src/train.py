import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import BASE_DIR, DATA_DIR

df = pd.read_csv(DATA_DIR / "clean.csv").dropna()

X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)
print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
}

best_name, best_acc, best_pipe = None, 0, None

for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                  max_df=0.9, max_features=50000)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n===== {name} =====")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, preds, target_names=["fake", "real"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    if acc > best_acc:
        best_name, best_acc, best_pipe = name, acc, pipe

print(f"\nBest model: {best_name} ({best_acc:.4f})")
joblib.dump(best_pipe, BASE_DIR / "models" / "best_model.joblib")

# Sanity check: what did Logistic Regression actually learn?
lr = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                              max_df=0.9, max_features=50000)),
    ("clf", LogisticRegression(max_iter=1000)),
]).fit(X_train, y_train)
names = np.array(lr.named_steps["tfidf"].get_feature_names_out())
order = np.argsort(lr.named_steps["clf"].coef_[0])
print("\nTop FAKE words:", list(names[order[:20]]))
print("Top REAL words:", list(names[order[-20:]]))