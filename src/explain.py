import numpy as np
import joblib
from load_data import BASE_DIR
from preprocess import clean_text

model = joblib.load(BASE_DIR / "models" / "best_model.joblib")
tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
names = np.array(tfidf.get_feature_names_out())


def explain(path, top=6):
    cleaned = clean_text(path.read_text(encoding="utf-8"))
    tokens = cleaned.split()
    known = sum(t in tfidf.vocabulary_ for t in tokens)

    vec = tfidf.transform([cleaned])
    idx = vec.nonzero()[1]
    contrib = vec.toarray()[0][idx] * clf.coef_[0][idx]
    order = np.argsort(contrib)
    score = contrib.sum() + clf.intercept_[0]

    print(f"\n{path.name}")
    print(f"  words after cleaning: {len(tokens)} | "
          f"in vocabulary: {known / max(len(tokens), 1):.0%}")
    print(f"  score {score:+.2f}  (bias {clf.intercept_[0]:+.2f})")
    print("  pushing FAKE:", list(names[idx[order[:top]]]))
    print("  pushing REAL:", list(names[idx[order[-top:]]][::-1]))


if __name__ == "__main__":
    for f in sorted((BASE_DIR / "samples").rglob("*.txt")):
        explain(f)