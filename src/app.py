import numpy as np
import joblib
import streamlit as st
from load_data import BASE_DIR
from preprocess import clean_text

UNCERTAIN_BAND = 0.25   # rough setting, NOT validated on a large set


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "best_model.joblib")


def top_words(model, cleaned, n=8):
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = np.array(tfidf.get_feature_names_out())
    vec = tfidf.transform([cleaned])
    idx = vec.nonzero()[1]
    contrib = vec.toarray()[0][idx] * clf.coef_[0][idx]
    order = np.argsort(contrib)
    return list(names[idx[order[:n]]]), list(names[idx[order[-n:]]][::-1])


model = load_model()

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("Fake News Detector")
st.caption("Trained on US political news (2016-2017). It reads writing "
           "style, not facts. Use it as a rough signal, not a fact-check.")

text = st.text_area("Paste an article or post", height=250)

if st.button("Analyze") and text.strip():
    cleaned = clean_text(text)
    if len(cleaned.split()) < 30:
        st.warning("Very short text: the result will be unreliable.")

    score = model.decision_function([cleaned])[0]
    if score > UNCERTAIN_BAND:
        st.success("Looks REAL")
    elif score < -UNCERTAIN_BAND:
        st.error("Looks FAKE")
    else:
        st.info("UNCERTAIN: the model can't tell")
    st.metric("Model score (negative = fake, positive = real)",
              f"{score:+.2f}")

    fake_w, real_w = top_words(model, cleaned)
    with st.expander("Which words drove this result?"):
        st.write("Pushing toward FAKE:", ", ".join(fake_w))
        st.write("Pushing toward REAL:", ", ".join(real_w))