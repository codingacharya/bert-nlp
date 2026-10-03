"""
NLP Fundamentals: Sentiment Analysis Pipeline on Amazon Reviews
Text preprocessing -> TF-IDF -> Word2Vec -> GloVe -> BERT basics

Run:  streamlit run sentiment_app.py
"""
import re
import random

import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Amazon Review Sentiment Pipeline", layout="wide")

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
POS = ["Great product, works perfectly", "Absolutely love it, highly recommend",
       "Excellent quality and fast delivery", "Very happy with this purchase",
       "Battery life is amazing", "Worth every penny", "Sturdy, well made and easy to use",
       "Exceeded my expectations", "Five stars, would buy again", "Fantastic value for money"]
NEG = ["Terrible product, stopped working in a week", "Do not buy, complete waste of money",
       "Very poor quality and awful customer service", "Not happy, it broke on day two",
       "Battery life is horrible", "Not worth the price", "Flimsy, cheap and hard to use",
       "Did not meet my expectations", "One star, would not buy again", "Disappointed, it arrived damaged"]
EXTRA = ["The packaging was fine.", "I ordered it last month.", "Shipping took five days.",
         "My family uses it daily.", "It looks like the picture."]


@st.cache_data(show_spinner=False)
def make_demo(n=600, seed=42):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        label = rng.choice([0, 1])
        core = rng.choice(POS if label else NEG)
        text = f"{core}. {rng.choice(EXTRA)} {rng.choice(POS if label else NEG)}."
        rows.append((text, label))
    return pd.DataFrame(rows, columns=["text", "label"])


@st.cache_data(show_spinner="Downloading Amazon Polarity from Hugging Face...")
def load_hf_amazon(n):
    from datasets import load_dataset
    ds = load_dataset("amazon_polarity", split="train").shuffle(seed=42).select(range(n))
    df = ds.to_pandas()
    df["text"] = df["title"].fillna("") + ". " + df["content"].fillna("")
    return df[["text", "label"]]  # label: 1 = positive, 0 = negative


def load_csv(file, text_col, rating_col):
    df = pd.read_csv(file)
    df = df[[text_col, rating_col]].dropna()
    df.columns = ["text", "rating"]
    df = df[df["rating"] != 3]                       # drop neutral
    df["label"] = (df["rating"] >= 4).astype(int)    # 4-5 positive, 1-2 negative
    return df[["text", "label"]]


# ----------------------------------------------------------------------------
# Text preprocessing
# ----------------------------------------------------------------------------
STEMMER = PorterStemmer()
NEGATIONS = {"no", "not", "nor", "never", "n't", "cannot"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS


def preprocess(text, lower=True, strip_html=True, strip_urls=True, strip_punct=True,
               rm_stop=True, stem=False):
    if strip_html:
        text = re.sub(r"<.*?>", " ", text)
    if strip_urls:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
    if lower:
        text = text.lower()
    text = re.sub(r"n't\b", " not", text)            # keep negation: "don't" -> "do not"
    if strip_punct:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    if rm_stop:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def evaluate(name, y_true, y_pred, extra=""):
    res = {"Model": name, "Accuracy": accuracy_score(y_true, y_pred),
           "F1 (macro)": f1_score(y_true, y_pred, average="macro"), "Notes": extra}
    st.session_state.setdefault("results", {})[name] = res
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{res['Accuracy']:.3f}")
    c2.metric("Macro F1", f"{res['F1 (macro)']:.3f}")
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                      index=["True: Neg", "True: Pos"], columns=["Pred: Neg", "Pred: Pos"])
    st.dataframe(cm)
    with st.expander("Classification report"):
        st.text(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))


def doc_vector(tokens, kv, dim):
    vecs = [kv[t] for t in tokens if t in kv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def embed_docs(texts, kv, dim):
    return np.vstack([doc_vector(t.split(), kv, dim) for t in texts])


@st.cache_resource(show_spinner="Loading GloVe vectors (first time ~130 MB)...")
def load_glove(name):
    import gensim.downloader as api
    return api.load(name)


@st.cache_resource(show_spinner="Loading BERT model...")
def load_bert(model_name):
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=256)


# ----------------------------------------------------------------------------
# Sidebar: data + split
# ----------------------------------------------------------------------------
st.title("Sentiment Analysis Pipeline on Amazon Reviews")
st.caption("Text preprocessing -> TF-IDF -> Word2Vec -> GloVe -> BERT")

with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Built-in demo", "Hugging Face: amazon_polarity", "Upload CSV"])
    if source == "Built-in demo":
        df = make_demo()
    elif source.startswith("Hugging"):
        n = st.slider("Number of reviews", 1000, 50000, 5000, 1000)
        df = load_hf_amazon(n)
    else:
        up = st.file_uploader("CSV file", type="csv")
        if up is None:
            st.info("Upload a CSV with a review text column and a 1-5 rating column.")
            st.stop()
        cols = pd.read_csv(up, nrows=1).columns.tolist()
        up.seek(0)
        text_col = st.selectbox("Text column", cols)
        rating_col = st.selectbox("Rating column (1-5)", cols, index=min(1, len(cols) - 1))
        df = load_csv(up, text_col, rating_col)

    st.header("2. Preprocessing")
    opt = dict(
        lower=st.checkbox("Lowercase", True),
        strip_html=st.checkbox("Remove HTML tags", True),
        strip_urls=st.checkbox("Remove URLs", True),
        strip_punct=st.checkbox("Remove punctuation/digits", True),
        rm_stop=st.checkbox("Remove stopwords (keep negations)", True),
        stem=st.checkbox("Porter stemming", False),
    )

    st.header("3. Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

df = df.drop_duplicates("text").reset_index(drop=True)
df["clean"] = df["text"].apply(lambda t: preprocess(t, **opt))
df = df[df["clean"].str.len() > 0]

X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
    df["clean"], df["label"], df["text"], test_size=test_size,
    random_state=42, stratify=df["label"])

tabs = st.tabs(["Data", "Preprocessing", "TF-IDF", "Word2Vec", "GloVe", "BERT",
                "Compare", "Try it"])

# ----------------------------------------------------------------------------
# Tab 0: Data
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", len(df))
    c2.metric("Positive", int((df.label == 1).sum()))
    c3.metric("Negative", int((df.label == 0).sum()))
    st.bar_chart(df["label"].map({0: "Negative", 1: "Positive"}).value_counts())
    st.dataframe(df[["text", "label"]].head(20), use_container_width=True)
    st.caption(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# ----------------------------------------------------------------------------
# Tab 1: Preprocessing
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Preprocessing, step by step")
    sample = st.text_area(
        "Try a review",
        "I DON'T like this <b>charger</b>!! It stopped working after 2 days. See http://x.com",
    )
    st.write("**Raw:**", sample)
    steps = [
        ("Remove HTML", dict(lower=False, strip_urls=False, strip_punct=False, rm_stop=False, stem=False)),
        ("+ URLs, lowercase", dict(strip_punct=False, rm_stop=False, stem=False)),
        ("+ Punctuation/digits removed", dict(rm_stop=False, stem=False)),
        ("+ Stopwords removed", dict(stem=False)),
        ("+ Stemming", dict()),
    ]
    for label, kw in steps:
        st.write(f"**{label}:**", preprocess(sample, **kw))
    st.write("**Tokens:**", preprocess(sample, **{**opt}).split())
    st.info("Negations (not, no, never) are kept because removing them flips sentiment "
            "('not good' -> 'good').")

# ----------------------------------------------------------------------------
# Tab 2: TF-IDF
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("TF-IDF + Logistic Regression")
    c1, c2, c3 = st.columns(3)
    ngram = c1.selectbox("N-grams", ["Unigrams", "Uni+Bigrams"])
    max_feat = c2.slider("Max features", 500, 20000, 5000, 500)
    C = c3.slider("Regularization C", 0.1, 10.0, 1.0, 0.1)
    st.latex(r"\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\cdot\log\frac{1+N}{1+\mathrm{df}(t)}+1")

    if st.button("Train TF-IDF model"):
        vec = TfidfVectorizer(max_features=max_feat,
                              ngram_range=(1, 1) if ngram == "Unigrams" else (1, 2))
        Xtr = vec.fit_transform(X_tr)
        Xte = vec.transform(X_te)
        clf = LogisticRegression(C=C, max_iter=1000).fit(Xtr, y_tr)
        st.session_state["tfidf"] = (vec, clf)
        evaluate("TF-IDF + LR", y_te, clf.predict(Xte), f"{ngram}, {max_feat} features")

        names = np.array(vec.get_feature_names_out())
        order = np.argsort(clf.coef_[0])
        a, b = st.columns(2)
        a.write("**Most negative terms**")
        a.dataframe(pd.DataFrame({"term": names[order[:15]], "weight": clf.coef_[0][order[:15]]}))
        b.write("**Most positive terms**")
        b.dataframe(pd.DataFrame({"term": names[order[-15:][::-1]],
                                  "weight": clf.coef_[0][order[-15:][::-1]]}))

# ----------------------------------------------------------------------------
# Tab 3: Word2Vec
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Word2Vec (trained on your reviews) + Logistic Regression")
    c1, c2, c3, c4 = st.columns(4)
    dim = c1.slider("Vector size", 50, 300, 100, 50)
    window = c2.slider("Window", 2, 10, 5)
    epochs = c3.slider("Epochs", 5, 50, 15, 5)
    arch = c4.selectbox("Architecture", ["Skip-gram", "CBOW"])
    st.caption("Document vector = average of its word vectors.")

    if st.button("Train Word2Vec model"):
        from gensim.models import Word2Vec
        sentences = [t.split() for t in X_tr]
        w2v = Word2Vec(sentences, vector_size=dim, window=window, min_count=2,
                       sg=1 if arch == "Skip-gram" else 0, epochs=epochs, workers=2, seed=42)
        st.session_state["w2v"] = w2v
        Xtr, Xte = embed_docs(X_tr, w2v.wv, dim), embed_docs(X_te, w2v.wv, dim)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, y_tr)
        st.session_state["w2v_clf"] = clf
        evaluate("Word2Vec + LR", y_te, clf.predict(Xte), f"{arch}, dim={dim}")
        st.write(f"Vocabulary size: {len(w2v.wv)}")

    if "w2v" in st.session_state:
        w = st.text_input("Find similar words to", "good", key="w2v_q")
        wv = st.session_state["w2v"].wv
        if w in wv:
            st.dataframe(pd.DataFrame(wv.most_similar(w, topn=10), columns=["word", "cosine"]))
        else:
            st.warning("Word not in vocabulary (try a frequent word).")

# ----------------------------------------------------------------------------
# Tab 4: GloVe
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("GloVe (pre-trained) + Logistic Regression")
    glove_name = st.selectbox("Pre-trained vectors",
                              ["glove-wiki-gigaword-50", "glove-wiki-gigaword-100",
                               "glove-wiki-gigaword-200", "glove-twitter-100"])
    st.caption("Word2Vec learns from local context windows; GloVe factorizes global "
               "co-occurrence counts. Here the vectors are pre-trained on Wikipedia/Twitter.")

    if st.button("Load GloVe and train"):
        kv = load_glove(glove_name)
        st.session_state["glove"] = kv
        d = kv.vector_size
        Xtr, Xte = embed_docs(X_tr, kv, d), embed_docs(X_te, kv, d)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, y_tr)
        st.session_state["glove_clf"] = clf
        evaluate("GloVe + LR", y_te, clf.predict(Xte), glove_name)

    if "glove" in st.session_state:
        g = st.text_input("Similar words / analogy: king - man + woman", "good", key="glove_q")
        kv = st.session_state["glove"]
        if g in kv:
            st.dataframe(pd.DataFrame(kv.most_similar(g, topn=10), columns=["word", "cosine"]))
        if st.checkbox("Show analogy: king - man + woman"):
            st.write(kv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))

# ----------------------------------------------------------------------------
# Tab 5: BERT
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("BERT basics")
    st.markdown(
        "- **WordPiece tokenization** splits rare words into sub-words.\n"
        "- **Contextual embeddings**: 'bank' gets a different vector in different sentences.\n"
        "- Here we use a **pre-trained, already fine-tuned** DistilBERT (SST-2) for inference. "
        "Fine-tuning on Amazon reviews would improve it further."
    )
    bert_name = st.selectbox("Model", ["distilbert-base-uncased-finetuned-sst-2-english",
                                       "nlptown/bert-base-multilingual-uncased-sentiment"])

    st.markdown("**Tokenization demo**")
    tok_text = st.text_input("Text", "The battery is unbelievably disappointing")
    if st.button("Show tokens"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        enc = tok(tok_text)
        st.write(pd.DataFrame({"token": tok.convert_ids_to_tokens(enc["input_ids"]),
                               "id": enc["input_ids"]}).T)

    st.markdown("**Evaluate on test set**")
    n_eval = st.slider("Test reviews to score (CPU is slow)", 50, 1000, 200, 50)
    if st.button("Run BERT inference"):
        pipe = load_bert(bert_name)
        texts = raw_te.iloc[:n_eval].tolist()
        with st.spinner("Scoring..."):
            out = pipe(texts, batch_size=16)
        if "nlptown" in bert_name:   # 1-5 stars
            preds = [1 if int(o["label"][0]) >= 4 else 0 for o in out]
        else:
            preds = [1 if o["label"] == "POSITIVE" else 0 for o in out]
        evaluate("BERT (DistilBERT)", y_te.iloc[:n_eval], preds, f"{bert_name}, n={n_eval}")

# ----------------------------------------------------------------------------
# Tab 6: Compare
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Model comparison")
    res = st.session_state.get("results", {})
    if res:
        table = pd.DataFrame(res.values()).set_index("Model")
        st.dataframe(table.style.format({"Accuracy": "{:.3f}", "F1 (macro)": "{:.3f}"}))
        st.bar_chart(table[["Accuracy", "F1 (macro)"]])
    else:
        st.info("Train at least one model in the earlier tabs.")

# ----------------------------------------------------------------------------
# Tab 7: Try it
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Predict a new review")
    review = st.text_area("Review text", "Stopped working after a week. Not worth the money.",
                          key="predict_text")
    if st.button("Predict"):
        clean = preprocess(review, **opt)
        rows = []
        if "tfidf" in st.session_state:
            vec, clf = st.session_state["tfidf"]
            p = clf.predict_proba(vec.transform([clean]))[0, 1]
            rows.append(("TF-IDF + LR", p))
        if "w2v_clf" in st.session_state:
            wv = st.session_state["w2v"].wv
            v = doc_vector(clean.split(), wv, wv.vector_size).reshape(1, -1)
            rows.append(("Word2Vec + LR", st.session_state["w2v_clf"].predict_proba(v)[0, 1]))
        if "glove_clf" in st.session_state:
            kv = st.session_state["glove"]
            v = doc_vector(clean.split(), kv, kv.vector_size).reshape(1, -1)
            rows.append(("GloVe + LR", st.session_state["glove_clf"].predict_proba(v)[0, 1]))
        try:
            pipe = load_bert("distilbert-base-uncased-finetuned-sst-2-english")
            o = pipe(review)[0]
            rows.append(("DistilBERT", o["score"] if o["label"] == "POSITIVE" else 1 - o["score"]))
        except Exception as e:
            st.caption(f"BERT unavailable: {e}")
        st.write("**Cleaned text:**", clean)
        if rows:
            out = pd.DataFrame(rows, columns=["Model", "P(positive)"])
            out["Prediction"] = np.where(out["P(positive)"] >= 0.5, "Positive", "Negative")
            st.dataframe(out.style.format({"P(positive)": "{:.3f}"}))
        else:
            st.info("Train a model first.")