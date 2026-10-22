import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from load_data import DATA_DIR

df = pd.read_csv(DATA_DIR / "clean.csv").dropna()

WEB = ["featured image", "getty images", "image via", "via", "image",
       "video", "watch", "read", "com", "getty", "screenshot",
       "twitter", "pic", "facebook"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "nov", "oct", "dec", "jan", "feb"]

variants = {
    "baseline": [],
    "without web artifacts": WEB,
    "without artifacts + weekdays/months": WEB + DAYS,
}

for name, words in variants.items():
    texts = df["content"]
    if words:
        pattern = r"\b(" + "|".join(words) + r")\b"
        texts = texts.str.replace(pattern, " ", regex=True)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, df["label"], test_size=0.2, random_state=42,
        stratify=df["label"],
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                  max_df=0.9, max_features=50000)),
        ("clf", LinearSVC()),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    