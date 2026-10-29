# NewsSieve
### Fake News Detection Using Machine Learning and Natural Language Processing

NewsSieve is an educational machine-learning project that classifies news article text according to patterns learned from a labelled dataset. It includes a Streamlit web application for analyzing text, reviewing influential words, and exploring evaluation results.

> **Important:** NewsSieve is a text-classification prototype, not a fact-checking system. It does not independently verify claims against trusted sources. A result such as **LOOKS REAL** is not proof that an article is true, and **LOOKS FAKE** is not proof that an article is false.

---

## Features

- **Text preprocessing:** lowercasing, URL and non-letter removal, selected Reuters-artifact cleanup, whitespace normalization, and English stopword removal.
- **TF-IDF features:** unigrams (single words) and bigrams (two-word sequences).
- **Supervised model comparison:** Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machine (Linear SVM).
- **Ensemble experiments:** Bagging and AdaBoost.
- **Evaluation tools:** accuracy, precision, recall, F1-score, confusion matrices, and a robustness experiment.
- **Additional coursework experiments:** K-means clustering and a separate Find-S concept-learning demonstration.
- **Streamlit app:** article analysis, an experimental uncertain band, influential-word explanations, session-only prediction history, and a model-performance dashboard.
- **Exploratory sample evaluation:** a script for evaluating text files in a local `samples/` directory.

## Technology Stack

- Python
- Pandas and NumPy
- scikit-learn
- NLTK
- Joblib
- Streamlit

---

## How It Works

1. **Load data:** load the fake and real news CSV files and assign the target labels.
2. **Prepare and clean text:** remove selected noise and unwanted text artifacts, normalize the text, remove empty records, and deduplicate articles.
3. **Split the data:** create a stratified 80/20 train-test split with `random_state=42`.
4. **Extract features:** use TF-IDF with unigrams and bigrams (`min_df=5`, `max_df=0.9`, and `max_features=50000`).
5. **Train and compare models:** fit the classifiers and evaluate them using the same held-out test split.
6. **Save the model:** save the selected TF-IDF/classifier pipeline with Joblib.
7. **Analyze text in the app:** load the saved pipeline, preprocess user input, and display a prediction, decision score, and influential terms.

The label mapping used by this project is:

- `0` = fake
- `1` = real

For the deployed Linear SVM, the decision score is **not a probability**. The app currently uses an experimental threshold band: scores above `+0.25` show **LOOKS REAL**, scores below `-0.25` show **LOOKS FAKE**, and scores between them show **UNCERTAIN**. This band has not been statistically calibrated.

---

## Experimental Results

The following figures were measured using a stratified 80/20 split of the cleaned dataset, with 30,911 training articles and 7,728 test articles. Precision, recall, and F1-score below refer to the **real-news class** (label `1`).

| Model | Accuracy | Precision (Real) | Recall (Real) | F1-score (Real) |
|---|---:|---:|---:|---:|
| Logistic Regression | 98.36% | 98.06% | 98.96% | 98.51% |
| Multinomial Naive Bayes | 95.41% | 96.59% | 94.97% | 95.78% |
| **Linear SVM** | **99.11%** | **99.10%** | **99.27%** | **99.19%** |
| Bagging | 96.80% | 97.34% | 96.81% | 97.08% |
| AdaBoost | 95.78% | 94.43% | 98.09% | 96.23% |

Linear SVM achieved the highest accuracy on this particular test split and is the model used by the app.

### Additional experiments

- **Robustness ablation:** baseline accuracy was 99.1071%; removing selected web artifacts gave 98.9260%; removing selected web and date artifacts gave 98.8354%. This tests only those selected terms and does not establish that the model understands factual content.
- **K-means clustering:** Adjusted Rand Index (ARI) = 0.1619 and Normalized Mutual Information (NMI) = 0.1174. These indicate limited agreement between the exploratory clusters and the reference labels.
- **Small sample evaluation:** 8 of 11 article samples matched labels inferred from their filenames (72.73%). The sample is small and the labels have not all been independently verified; this is **not** a reliable estimate of real-world accuracy.

### Interpreting the results responsibly

The original dataset mainly represents US news from 2016–2017. A random held-out split from the same dataset can reward source-specific wording, article style, topic patterns, or other dataset correlations. Performance on recent news, Indian news, unfamiliar publishers, or different content types may be lower. Satire, fact-check articles, and fabricated news should not automatically be treated as the same category.

---

## Repository Structure

```text
news-checker-agent/
├── data/                  # Local dataset files (not necessarily committed)
├── models/                # Saved model pipeline (generated locally)
├── samples/               # Optional local .txt sample articles
├── results/                # Evaluation CSV/JSON outputs
└── src/
    ├── app.py
    ├── load_data.py
    ├── explore_data.py
    ├── preprocess.py
    ├── train.py
    ├── robustness.py
    ├── predict.py
    ├── explain.py
    ├── evaluate_models.py
    ├── evaluate_real_world.py
    ├── bagging_model.py
    ├── adaboost_model.py
    ├── kmeans_clustering.py
    └── find_s.py
```

Some files and folders are created locally by running the scripts. If a dataset, saved model, sample article, or evaluation output is not included in the repository, generate it locally as described below.

---

## Setup (Windows / PowerShell)

### 1. Clone the repository

```powershell
git clone https://github.com/Musadiq-khan/news-checker-agent.git
cd news-checker-agent
```

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use the environment's Python executable directly instead of changing system-wide execution policy.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn nltk joblib streamlit matplotlib seaborn
```

### 4. Add the dataset locally

Place the original dataset files in the project’s `data/` folder:

```text
data/
├── Fake.csv
└── True.csv
```

The project uses the Kaggle Fake and Real News Dataset, often referred to as the ISOT Fake and Real News Dataset. Follow the dataset's own licence and usage terms. The dataset files are not required to be committed to GitHub.

### 5. Prepare the cleaned data and train the model

Run from the project root:

```powershell
python src/preprocess.py
python src/train.py
```

These commands generate `data/clean.csv` and `models/best_model.joblib`. The preprocessing script downloads NLTK English stopwords if they are not already available, so internet access may be needed on the first run.

### 6. Generate evaluation results

```powershell
python src/evaluate_models.py
python src/robustness.py
```

To run the exploratory evaluation on local text samples, place `.txt` files in `samples/`. A filename beginning with `fake_` or `real_` supplies a **provisional expected label** to the evaluation script:

```powershell
python src/evaluate_real_world.py
```

Review each label before interpreting the sample metrics.

### 7. Launch the web application

```powershell
python -m streamlit run src/app.py
```

Streamlit will print a local URL (normally `http://localhost:8501`) to open in a browser.

---

## Generated Evaluation Files

The evaluation scripts can create the following local files:

- `results/model_metrics.csv` — comparison metrics for the five evaluated models.
- `results/evaluation_details.json` — classification reports and confusion matrices.
- `results/real_world_evaluation.csv` — per-article exploratory sample predictions.
- `results/real_world_evaluation_summary.json` — summary of the small sample evaluation.

The dashboard requires the corresponding result files to be present locally. Run the relevant evaluation scripts to recreate missing files.

## Limitations and Future Work

Current limitations include dataset age and coverage, source/style bias, the very small external sample, provisional filename-derived labels, English-only preprocessing, and an uncalibrated SVM decision-score threshold. The system has no retrieval or evidence-verification component.

Potential next steps include building a larger independently labelled test set, evaluating on publishers or time periods held out from training, calibrating the uncertain band on a separate validation set, and adding evidence retrieval with transparent source links.

## Responsible Use

Use NewsSieve as a learning and text-analysis aid only. Do not use its output alone to accuse a source, label a person, censor information, or make consequential decisions. Verify important claims using multiple reliable sources.

## Acknowledgements

This project uses the publicly available Fake and Real News dataset and open-source Python libraries, particularly scikit-learn, Pandas, NumPy, NLTK, Joblib, and Streamlit. Please consult the respective project and dataset pages for citation and licence requirements.

## Project Repository

[https://github.com/Musadiq-khan/news-checker-agent](https://github.com/Musadiq-khan/news-checker-agent)
