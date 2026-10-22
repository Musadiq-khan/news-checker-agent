import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data

df = load_data()

# 1. Duplicates
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate texts:", df["text"].duplicated().sum())

# 2. Empty texts and article length
print("Empty texts:", (df["text"].str.strip() == "").sum())
df["word_count"] = df["text"].str.split().str.len()
print(df.groupby("label")["word_count"].describe())

# 3. Subject vs label
print(pd.crosstab(df["subject"], df["label"]))

# 4. Leakage check: publisher tag
df["has_reuters"] = df["text"].str.contains("Reuters", case=False)
print(df.groupby("label")["has_reuters"].mean())

# 5. Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x="label", data=df, ax=axes[0])
axes[0].set_title("Class balance (0 = fake, 1 = real)")
sns.histplot(data=df[df["word_count"] < 2000], x="word_count",
             hue="label", bins=50, ax=axes[1])
axes[1].set_title("Article length by class")
plt.tight_layout()
plt.show()