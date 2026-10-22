import joblib
from load_data import BASE_DIR
from preprocess import clean_text

model = joblib.load(BASE_DIR / "models" / "best_model.joblib")


def predict(text):
    cleaned = clean_text(text)
    score = model.decision_function([cleaned])[0]
    label = "REAL" if score > 0 else "FAKE"
    return label, score


if __name__ == "__main__":
    folder = BASE_DIR / "samples"
    correct = total = 0
    for f in sorted(folder.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        label, score = predict(text)
        expected = f.name.split("_")[0].upper()
        ok = label == expected
        correct += ok
        total += 1
        print(f"{'OK  ' if ok else 'MISS'} {f.name:38s} "
              f"predicted={label:4s} score={score:+.2f}")
    print(f"\n{correct}/{total} correct")