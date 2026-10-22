import re
import nltk
from nltk.corpus import stopwords
from load_data import load_data, DATA_DIR

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def remove_leaks(text):
    # Removes datelines like "WASHINGTON (Reuters) - " at the start
    text = re.sub(r"^.{0,100}?\(Reuters\)\s*-?\s*", "", text)
    # Removes any other mention of the word
    text = re.sub(r"reuters", " ", text, flags=re.IGNORECASE)
    return text


def clean_text(text):
    text = remove_leaks(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)           # keep letters only
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


if __name__ == "__main__":
    df = load_data()
    print("Before:", df.shape)
    print(df["label"].value_counts())

    df = df.drop(columns=["subject", "date"])
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset="text")

    df["content"] = df["title"] + " " + df["text"]

    sample = df["content"].iloc[0]
    df["content"] = df["content"].apply(clean_text)
    print("\nBEFORE:", sample[:300])
    print("AFTER: ", df["content"].iloc[0][:300])

    df = df[df["content"].str.split().str.len() > 0]
    print("\nAfter cleaning:", df.shape)
    print(df["label"].value_counts())
    print("Still contains 'reuters':",
          df["content"].str.contains("reuters").sum())

    df[["content", "label"]].to_csv(DATA_DIR / "clean.csv", index=False)
    print("Saved data/clean.csv")