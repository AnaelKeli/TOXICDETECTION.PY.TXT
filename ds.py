# group12_fasttext_3class_app.py — Toxic/Offensive/Hateful (3-class, multi‑label) using FastText
# --------------------------------------------------------------------------------------------
# Pages: Overview • Dataset • Modeling & Evaluation • Predict • Model Comparison • Conclusion
# Embedding: FastText (gensim)
# Models: Logistic Regression & Random Forest (trained per label)
# Targets: three independent labels → is_toxic, is_offensive, is_hateful
# Metrics: per‑label Precision, Recall, F1, ROC‑AUC + Confusion Matrices + ROC Curves

import warnings
warnings.filterwarnings("ignore")

import re, numpy as np, pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ---- REQUIRED: FastText / gensim ----
try:
    from gensim.models import FastText
    from gensim.utils import simple_preprocess
except Exception:
    st.set_page_config(page_title="Group 12 • 3‑Class Comment Classifier", page_icon="🛡️", layout="wide")
    st.error(
        "This app requires **gensim** for FastText.\n\nInstall with: `pip install gensim`"
    )
    st.stop()

# Optional NLTK stopwords (best effort)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP = set(stopwords.words('english'))
except Exception:
    STOP = set()

# --------------------------------- Aesthetics ---------------------------------
st.set_page_config(page_title="Group 12 • 3‑Class Comment Classifier", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
:root{ --brand:#0f4c81; --bg:#f7fafc; --card:#ffffff; --muted:#64748b; --ink:#0f172a; }
.main { background: var(--bg); }
section[data-testid="stSidebar"] { background: #0f4c8112; }
h1,h2,h3 { font-weight: 800; letter-spacing:.2px; }
.hero { padding:1rem 1.25rem; background:linear-gradient(90deg,#0f4c81 0%,#1d75b3 100%);
        color:#fff; border-radius:16px; box-shadow:0 10px 30px rgba(15,76,129,.25); }
.hero h1 { margin:0 0 .25rem 0; font-size:2rem; }
.hero p { margin:.25rem 0 0 0; opacity:.95; }
.metric { background:var(--card); border:1px solid #e5e7eb; border-radius:14px; padding:14px; text-align:center; }
.metric h4{ margin:.1rem 0; color:var(--muted); font-weight:700;}
.metric div{ font-size:1.2rem; font-weight:800; color:var(--ink);}
.stButton>button{ background:var(--brand); color:#fff; border:0; border-radius:12px; padding:.6rem 1rem; font-weight:700; }
.stButton>button:hover{ filter:brightness(1.07); }
</style>
""", unsafe_allow_html=True)

# --------------------------- Session-state defaults ---------------------------
DEFAULTS = {
    "df": None,
    "df_balanced": None,   # holds post-preview balanced data
    "fasttext_model": None,     # gensim FastText model
    "fasttext_dim": 100,
    "trained_models": {},       # {label: {model_name: estimator}}
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------- Helpers ---------------------------
RAW_COLS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
TARGETS = ['is_toxic','is_offensive','is_hateful']


def clean_text(x: str) -> str:
    if pd.isna(x): return ''
    t = str(x).lower()
    t = re.sub(r'http\S+|www\S+|https\S+', ' ', t)   # URLs
    t = re.sub(r'<.*?>', ' ', t)                       # HTML
    t = re.sub(r'[^a-zA-Z\s]', ' ', t)                # keep letters/spaces only
    t = re.sub(r'\s+', ' ', t).strip()
    if STOP:
        toks = [w for w in t.split() if w not in STOP and len(w) > 2]
        return ' '.join(toks)
    return t


def build_three_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Map Jigsaw labels → three independent targets.
    toxic      := toxic OR severe_toxic
    offensive  := insult OR obscene  (coarse proxy)
    hateful    := identity_hate
    If raw columns are missing, try existing boolean/int columns with same names.
    """
    cols = set(df.columns.str.lower())
    def col(name):
        return name in cols

    # normalize column names for safe access
    df2 = df.copy()
    lower_map = {c: c.lower() for c in df2.columns}
    df2.rename(columns=lower_map, inplace=True)

    # fallbacks if any raw column is missing
    def getcol(name):
        return df2[name] if name in df2.columns else pd.Series([0]*len(df2), index=df2.index)

    toxic = (getcol('toxic') | getcol('severe_toxic')).astype(int)
    offensive = (getcol('insult') | getcol('obscene')).astype(int)
    hateful = getcol('identity_hate').astype(int)

    df2['is_toxic'] = toxic
    df2['is_offensive'] = offensive
    df2['is_hateful'] = hateful
    return df2


def ensure_two_classes_per_label(X: pd.Series, Y: pd.DataFrame):
    """Ensure each label has 0/1 present; if a label is all one class, raise a clear error.
       Light upsampling for very tiny minority classes per label.
    """
    X = pd.Series(X).reset_index(drop=True)
    Y = pd.DataFrame(Y).reset_index(drop=True)
    keep = []
    for col in Y.columns:
        vc = Y[col].value_counts()
        if Y[col].nunique() < 2:
            raise ValueError(f"Label '{col}' has a single class only. Include both positive and negative examples.")
        # per-label light upsample if minority is tiny (to stabilize stratified split)
        if vc.min() < 5:
            df_tmp = pd.DataFrame({'text': X, 'label': Y[col]})
            maj = vc.idxmax(); minc = vc.idxmin()
            df_min = df_tmp[df_tmp.label == minc].sample(n=max(10, vc[minc]*3), replace=True, random_state=42)
            df_bal = pd.concat([df_tmp[df_tmp.label == maj], df_min], ignore_index=True).sample(frac=1.0, random_state=42)
            X = df_bal['text']
            Y[col] = df_bal['label']
        keep.append(col)
    return X, Y[keep]


def fasttext_doc_vec(ft: FastText, text: str, dim: int) -> np.ndarray:
    if pd.isna(text) or not str(text).strip(): return np.zeros(dim)
    toks = simple_preprocess(str(text), deacc=True)
    vecs = [ft.wv[w] for w in toks if w in ft.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def make_fasttext(train_texts, test_texts, dim=100, window=5, min_count=2, sg=1, epochs=10):
    # Normalize inputs
    train_texts = pd.Series(train_texts).fillna("").astype(str)
    test_texts  = pd.Series(test_texts).fillna("").astype(str)

    # Tokenize training texts into sentences (lists of tokens)
    sents = [simple_preprocess(t, deacc=True) for t in train_texts]
    sents = [s for s in sents if len(s) > 0]
    if len(sents) == 0:
        raise ValueError("No valid tokens in training data after cleaning. Check your input or lower min_count.")

    ft = FastText(vector_size=dim, window=window, min_count=min_count, workers=4, sg=sg)
    ft.build_vocab(corpus_iterable=sents, update=False)
    if len(ft.wv) == 0:
        if min_count > 1:
            st.info(f"FastText: vocabulary empty with min_count={min_count}; retrying with min_count=1.")
            ft = FastText(vector_size=dim, window=window, min_count=1, workers=4, sg=sg)
            ft.build_vocab(corpus_iterable=sents, update=False)
        if len(ft.wv) == 0:
            raise ValueError("FastText vocabulary is empty even with min_count=1. Text may be too short after cleaning.")

    ft.train(corpus_iterable=sents, total_examples=len(sents), epochs=epochs)

    Xtr = np.vstack([fasttext_doc_vec(ft, t, ft.vector_size) for t in train_texts])
    Xte = np.vstack([fasttext_doc_vec(ft, t, ft.vector_size) for t in test_texts])
    return Xtr, Xte, ft

# --- Models & evaluation (FastText vectors -> classic ML) ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve


def evaluate_per_label(est, Xte, yte, thresh=0.5):
    prob = est.predict_proba(Xte)[:,1]
    pred = (prob >= thresh).astype(int)
    metrics = {
        'precision': precision_score(yte, pred, zero_division=0),
        'recall':    recall_score(yte, pred, zero_division=0),
        'f1':        f1_score(yte, pred, zero_division=0),
        'roc_auc':   roc_auc_score(yte, prob)
    }
    cm = confusion_matrix(yte, pred)
    fpr, tpr, _ = roc_curve(yte, prob)
    return metrics, cm, (fpr, tpr), prob


# ---------------------- Sample data (with 3 targets) ----------------------

def demo_dataframe():
    data = {
        'comment_text': [
            "This is a great video, thanks for sharing!",
            "You are so stupid and worthless",
            "I disagree with your opinion but respect your view",
            "Go kill yourself, nobody likes you",
            "Amazing content, keep up the good work!",
            "What an idiot, this is garbage",
            "Thanks for the tutorial, very helpful",
            "You're a pathetic loser",
            "Nice explanation, learned something new",
            "Hate this channel, waste of time"
        ],
        # Proxy raw labels to derive three targets
        'toxic': [0,1,0,1,0,1,0,1,0,1],
        'severe_toxic':[0,0,0,1,0,0,0,0,0,0],
        'obscene':[0,1,0,1,0,1,0,1,0,0],
        'threat':[0,0,0,1,0,0,0,0,0,0],
        'insult':[0,1,0,1,0,1,0,1,0,0],
        'identity_hate':[0]*10
    }
    df = pd.DataFrame(data)
    df = build_three_targets(df)
    return df


# ----------------------- Balancing utilities -----------------------
from typing import Literal

BalanceMethod = Literal['label-combos','per-label']

def _combo_key(df: pd.DataFrame) -> pd.Series:
    return df[TARGETS].astype(int).astype(str).agg('-'.join, axis=1)

# ---------- UPSAMPLING (existing) ----------

def balance_by_label_combos(df: pd.DataFrame, upsample_to: Literal['max','median']='max') -> pd.DataFrame:
    """Upsample each (is_toxic, is_offensive, is_hateful) combination to the same size.
    Keeps multilabel structure intact. """
    df = df.copy()
    key = _combo_key(df)
    groups = df.groupby(key)
    sizes = groups.size()
    if upsample_to == 'median':
        target_n = int(sizes.median()) if len(sizes) else 0
    else:
        target_n = int(sizes.max()) if len(sizes) else 0
    parts = []
    for _, g in groups:
        if len(g) == 0:
            continue
        if len(g) < target_n:
            extra = g.sample(n=target_n - len(g), replace=True, random_state=42)
            parts.append(pd.concat([g, extra], ignore_index=True))
        else:
            parts.append(g.sample(n=target_n, replace=False, random_state=42))
    out = pd.concat(parts, ignore_index=True) if parts else df
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def balance_per_label(df: pd.DataFrame, desired_ratio: float = 1.0) -> pd.DataFrame:
    """Simple per-label oversampling: for each label, oversample minority to reach ratio.
    Union of oversampled indices across labels, then deduplicate."""
    rng = np.random.default_rng(42)
    keep_idx = set()
    for lab in TARGETS:
        pos_idx = df.index[df[lab] == 1].tolist()
        neg_idx = df.index[df[lab] == 0].tolist()
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            continue
        target_pos = min(n_neg, int(desired_ratio * n_neg))
        if n_pos < target_pos:
            need = target_pos - n_pos
            sampled = rng.choice(pos_idx, size=need, replace=True)
            keep_idx.update(sampled.tolist())
        keep_idx.update(pos_idx)
        keep_idx.update(rng.choice(neg_idx, size=min(n_neg, target_pos), replace=False).tolist())
    if not keep_idx:
        return df.copy()
    out = df.loc[sorted(keep_idx)].copy()
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)

# ---------- DOWNSAMPLING (new) ----------

def downsample_by_label_combos(df: pd.DataFrame, downsample_to: Literal['min','median']='min') -> pd.DataFrame:
    """Downsample each label-combo group to the same size to REDUCE dataset size.
    downsample_to='min' → trims all combos to the smallest group size (maximal reduction).
    downsample_to='median' → trims larger groups to the median size (gentler reduction)."""
    df = df.copy()
    key = _combo_key(df)
    groups = df.groupby(key)
    sizes = groups.size()
    if len(sizes) == 0:
        return df
    target_n = int(sizes.min() if downsample_to=='min' else sizes.median())
    parts = []
    for _, g in groups:
        if len(g) >= target_n:
            parts.append(g.sample(n=target_n, replace=False, random_state=42))
        else:
            # group already below target; keep as-is (can't downsample further)
            parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def downsample_per_label(df: pd.DataFrame, max_neg_pos_ratio: float = 1.0) -> pd.DataFrame:
    """Per-label random undersampling of negatives to meet a maximum NEG:POS ratio.
    Example: max_neg_pos_ratio=1.0 → at most 1 negative per positive for each label.
    We union the kept indices across labels to preserve multi-label structure without exploding size."""
    rng = np.random.default_rng(42)
    keep_idx = set()
    for lab in TARGETS:
        pos_idx = df.index[df[lab] == 1].tolist()
        neg_idx = df.index[df[lab] == 0].tolist()
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0:
            # no positives; keep a small sample of negatives to avoid dropping the label entirely
            sample_n = min(200, n_neg)  # cap to avoid huge datasets
            keep_idx.update(rng.choice(neg_idx, size=sample_n, replace=False).tolist())
            continue
        target_neg = int(min(n_neg, max_neg_pos_ratio * n_pos))
        keep_idx.update(pos_idx)
        if target_neg > 0:
            keep_idx.update(rng.choice(neg_idx, size=target_neg, replace=False).tolist())
    if not keep_idx:
        return df.copy()
    out = df.loc[sorted(keep_idx)].copy()
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)

# --------------------------------- PAGES ---------------------------------

def page_overview():
    st.markdown("""
<div class="hero">
  <h1>Group 12 • 3‑Class Comment Classification</h1>
  <p>Predict <b>toxic</b>, <b>offensive</b>, and <b>hateful</b> language with FastText embeddings and classic ML models.</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="metric"><h4>Embedding</h4><div>FastText</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric"><h4>Models</h4><div>LogReg & RF (per label)</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric"><h4>Metrics</h4><div>Precision • Recall • F1 • AUC</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric"><h4>Targets</h4><div>Toxic • Offensive • Hateful</div></div>', unsafe_allow_html=True)

    st.markdown("### Workflow")
    st.markdown("**Dataset → Modeling & Evaluation → Predict → Model Comparison → Conclusion**")


def page_dataset():
    st.subheader("Dataset")
    c1, c2 = st.columns([2,1])
    with c1:
        up = st.file_uploader("Upload Jigsaw‑style CSV (must include `comment_text`)", type=['csv'])
    with c2:
        if st.button("Use Sample Data", use_container_width=True):
            st.session_state["df"] = demo_dataframe()
            st.session_state["df_balanced"] = None

    if up is not None:
        try:
            st.session_state["df"] = pd.read_csv(up)
            st.session_state["df_balanced"] = None
            st.success("File loaded!")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.session_state["df"] is None:
        st.info("Upload a CSV or click **Use Sample Data**.")
        return

    df = st.session_state["df"].copy()
    if "comment_text" not in df.columns:
        st.error("CSV must include a text column named `comment_text`.")
        return

    # Clean text + build three targets
    df["comment_text_clean"] = df["comment_text"].apply(clean_text)
    df = build_three_targets(df)

    # Quick stats
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric"><h4>Rows</h4><div>{len(df):,}</div></div>', unsafe_allow_html=True)
    with c2:
        rates = {k:int(df[k].sum()) for k in TARGETS}
        st.markdown(f'<div class="metric"><h4>Positives</h4><div>{rates}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric"><h4>Targets</h4><div>{TARGETS}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Preview (first 12 rows)")
    st.dataframe(df.head(12), use_container_width=True)

    present = [c for c in RAW_COLS if c in df.columns]
    if present:
        st.markdown("#### Raw label coverage (from Jigsaw)")
        st.bar_chart(df[present].sum())

    # ---------------- Balancing controls ----------------
    st.markdown("### Balance dataset")
    method = st.radio("Method", ["label-combos", "per-label"], index=0, horizontal=True)
    mode = st.radio("Sampling mode", ["Downsample (reduce size)", "Upsample (increase)"] , index=0, horizontal=True)

    if mode.startswith("Downsample"):
        colb1, colb2 = st.columns([1,1])
        with colb1:
            down_to = st.selectbox("Downsample to (combos)", ["min","median"], index=0,
                                   help="For label-combo downsampling: trim each combo to this size target.")
        with colb2:
            max_ratio = st.number_input("Max NEG:POS ratio (per-label)", min_value=0.2, max_value=10.0, value=1.0, step=0.1,
                                        help="For per-label downsampling: limit negatives per positive to this ratio.")
    else:
        colb1, colb2 = st.columns([1,1])
        with colb1:
            up_to = st.selectbox("Upsample to (combos)", ["max","median"], index=0,
                                 help="For label-combo upsampling: grow each combo to this size target.")
        with colb2:
            desired_ratio = st.number_input("POS:NEG ratio (per-label)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                            help="For per-label upsampling: target positive:negative ratio.")

    if st.button("Apply balancing", type="primary"):
        if mode.startswith("Downsample"):
            if method == 'label-combos':
                df_bal = downsample_by_label_combos(df, downsample_to=down_to)
            else:
                df_bal = downsample_per_label(df, max_neg_pos_ratio=max_ratio)
        else:
            if method == 'label-combos':
                df_bal = balance_by_label_combos(df, upsample_to=up_to)
            else:
                df_bal = balance_per_label(df, desired_ratio=desired_ratio)
        st.session_state["df_balanced"] = df_bal
        st.success(f"Balanced! Rows: {len(df_bal):,} (was {len(df):,})")
        # Show distribution by label combos
        key_before = df[TARGETS].astype(int).astype(str).agg('-'.join, axis=1).value_counts().rename('count_before')
        key_after = df_bal[TARGETS].astype(int).astype(str).agg('-'.join, axis=1).value_counts().rename('count_after')
        dist = pd.concat([key_before, key_after], axis=1).fillna(0).astype(int)
        st.write("Label‑combo distribution before/after:")
        st.dataframe(dist)

    st.session_state["df"] = df



def page_modeling():
    st.subheader("Modeling & Evaluation")

    base = st.session_state.get("df_balanced")
    if base is None:
        base = st.session_state.get("df")
    if base is None or "comment_text_clean" not in base.columns:
        st.warning("Go to **Dataset** and upload/prepare your data first. Use **Balance now** if you want a balanced set.")
        return
    df = base
    X_raw = df['comment_text_clean']
    Y_raw = df[TARGETS]

    try:
        X_bal, Y_bal = ensure_two_classes_per_label(X_raw, Y_raw)
    except ValueError as e:
        st.error(str(e)); return

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    from sklearn.model_selection import train_test_split
    Xtr_txt, Xte_txt, Ytr, Yte = train_test_split(X_bal, Y_bal, test_size=test_size,
                                                 random_state=42, stratify=Y_bal.astype(str).agg('-'.join, axis=1))

    # FastText hyperparameters
    colA, colB, colC, colD = st.columns(4)
    dim   = colA.slider("FT dimension", 50, 300, 100, 50)
    win   = colB.slider("FT window", 2, 10, 5)
    minc  = colC.slider("Min word count", 1, 10, 2)
    algo  = colD.selectbox("Algorithm", ["Skip-gram (sg=1)", "CBOW (sg=0)"], index=0)
    sg    = 1 if algo.startswith("Skip") else 0
    epochs= st.slider("Epochs", 5, 30, 10)

    with st.spinner("Training FastText & building document vectors..."):
        try:
            Xtr, Xte, ft = make_fasttext(Xtr_txt, Xte_txt, dim=dim, window=win, min_count=minc, sg=sg, epochs=epochs)
        except Exception as e:
            st.error(f"FastText failed: {e}")
            return
        st.session_state["fasttext_model"] = ft
        st.session_state["fasttext_dim"] = dim
        st.success(f"FastText ready: train {Xtr.shape}, test {Xte.shape}")

    # Train models per label on FastText vectors
    trained = {}
    with st.spinner("Training models (per label)..."):
        for label in TARGETS:
            lr = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced').fit(Xtr, Ytr[label])
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, Ytr[label])
            trained[label] = {"Logistic Regression": lr, "Random Forest": rf}
    st.session_state["trained_models"] = trained

    # Evaluate per label and per model
    all_rows = []
    rocs = {label: {} }  # nested dict label->model->(fpr,tpr)
    rocs = {label:{} for label in TARGETS}

    for label in TARGETS:
        for name, est in trained[label].items():
            m, cm, roc, _ = evaluate_per_label(est, Xte, Yte[label])
            all_rows.append({"label": label, "model": name, **m})
            rocs[label][name] = roc

    dfm = pd.DataFrame(all_rows).set_index(['label','model']).round(4)

    st.markdown("#### Test metrics (per label)")
    st.dataframe(dfm, use_container_width=True)

    # Confusion matrices
    st.markdown("#### Confusion matrices")
    for label in TARGETS:
        st.markdown(f"**{label}**")
        for name, est in trained[label].items():
            _, cm, _, _ = evaluate_per_label(est, Xte, Yte[label])
            st.markdown(f"{name}")
            st.write(pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1']))

    # ROC curves per label
    st.markdown("#### ROC curves")
    for label in TARGETS:
        fig, ax = plt.subplots(figsize=(5.5,4.5))
        for name, (fpr, tpr) in rocs[label].items():
            auc = float(dfm.loc[(label, name), 'roc_auc'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0,1],[0,1],'k--', alpha=0.6)
        ax.set_title(f"ROC • {label}")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)


def page_predict():
    st.subheader("Predict")
    if not st.session_state["trained_models"]:
        st.warning("Train models in **Modeling & Evaluation** first.")
        return

    label = st.selectbox("Target label", TARGETS, index=0)
    model_names = list(st.session_state["trained_models"][label].keys())

    tab1, tab2 = st.tabs(["Single comment", "Batch CSV"])

    with tab1:
        text = st.text_area("Enter a comment", height=120, placeholder="Type/paste a comment…")
        model_sel = st.selectbox("Model", model_names, index=0)
        thr = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
        if st.button("Predict"):
            ft = st.session_state["fasttext_model"]; dim = st.session_state["fasttext_dim"]
            Xone = np.vstack([fasttext_doc_vec(ft, clean_text(text), dim)])
            p = st.session_state["trained_models"][label][model_sel].predict_proba(Xone)[:,1][0]
            pred = int(p >= thr)
            st.markdown(f"**Probability ({label}):** {p:.3f} → **Pred:** {'🚨 POSITIVE' if pred==1 else '✅ NEGATIVE'}")

    with tab2:
        up2 = st.file_uploader("Upload CSV for batch predictions (must contain a text column)", type=['csv'])
        if up2 is not None:
            dfb = pd.read_csv(up2)
            # pick a likely text column
            text_col = None
            for c in ['comment_text','comment','text','content']:
                if c in dfb.columns: text_col = c; break
            if text_col is None:
                st.error("No valid text column found.")
            else:
                ft = st.session_state["fasttext_model"]; dim = st.session_state["fasttext_dim"]
                thr_b = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05, key="thr_b")
                # compute probabilities for all three labels using chosen models
                probs = {}
                preds = {}
                for lab in TARGETS:
                    # always use the first model (LogReg) by default for batch; or allow selection per label
                    est = st.session_state["trained_models"][lab]["Logistic Regression"]
                    Xb = np.vstack([fasttext_doc_vec(ft, clean_text(t), dim) for t in dfb[text_col].fillna("").astype(str)])
                    pr = est.predict_proba(Xb)[:,1]
                    probs[lab] = pr
                    preds[lab] = (pr >= thr_b).astype(int)

                out = dfb.copy()
                for lab in TARGETS:
                    out[f'{lab}_prob'] = probs[lab]
                    out[f'{lab}_pred'] = preds[lab]
                st.dataframe(out.head(20), use_container_width=True)
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV",
                                   data=csv,
                                   file_name=f"threeclass_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")


def page_compare():
    st.subheader("Model Comparison")
    if not st.session_state["trained_models"]:
        st.info("Train models on **Modeling & Evaluation** first.")
        return

    # Quick side-by-side metric table on a fresh split (per label)
    df = st.session_state.get("df_balanced")
    if df is None:
        df = st.session_state.get("df")
    if df is None or "comment_text_clean" not in df.columns:
        st.warning("Go back to **Dataset** and prepare data.")
        return

    X_raw = df['comment_text_clean']
    Y_raw = df[TARGETS]

    try:
        X_bal, Y_bal = ensure_two_classes_per_label(X_raw, Y_raw)
    except ValueError as e:
        st.error(str(e)); return

    from sklearn.model_selection import train_test_split
    Xtr_txt, Xte_txt, Ytr, Yte = train_test_split(X_bal, Y_bal, test_size=0.2, random_state=42,
                                                  stratify=Y_bal.astype(str).agg('-'.join, axis=1))

    ft = st.session_state["fasttext_model"]; dim = st.session_state["fasttext_dim"]
    Xte = np.vstack([fasttext_doc_vec(ft, t, dim) for t in Xte_txt])

    rows = []
    for label in TARGETS:
        for name, est in st.session_state["trained_models"][label].items():
            m, _, _, _ = evaluate_per_label(est, Xte, Yte[label])
            rows.append({"label": label, "model": name, **m})
    dfm = pd.DataFrame(rows).set_index(["label","model"]).round(4)
    st.dataframe(dfm, use_container_width=True)

    # Bar chart per metric (macro avg by label)
    st.markdown("#### Macro averages by label (LogReg)")
    macro_rows = []
    for label in TARGETS:
        est = st.session_state["trained_models"][label]["Logistic Regression"]
        m, _, _, _ = evaluate_per_label(est, Xte, Yte[label])
        macro_rows.append({"label": label, **m})
    dfa = pd.DataFrame(macro_rows).set_index('label').round(4)
    fig, ax = plt.subplots(figsize=(6,4))
    dfa[['precision','recall','f1','roc_auc']].plot(kind='bar', ax=ax)
    ax.set_ylim(0,1); ax.set_title("Metric comparison by label (LogReg)")
    st.pyplot(fig)


def page_conclusion():
    st.subheader("Conclusion & Next Steps")
    st.markdown(
        "- We reframed the problem as **three independent binary tasks** (multi‑label): toxic, offensive, hateful.\n"
        "- Mapping used: `toxic := toxic ∪ severe_toxic`, `offensive := insult ∪ obscene`, `hateful := identity_hate`.\n"
        "- Consider refining the mapping if you have a dataset with explicit 'offensive' labels, or include `threat` if you want it treated as offensive.\n"
        "- Next: threshold tuning per label, calibrating probabilities, class‑weighted training, or swapping in sentence transformers."
    )
    st.markdown("**Thanks!** 🎯")

# ----------------------------- Navigation -----------------------------
PAGES = {
    "Overview": page_overview,
    "Dataset": page_dataset,
    "Modeling & Evaluation": page_modeling,
    "Predict": page_predict,
    "Model Comparison": page_compare,
    "Conclusion": page_conclusion,
}
choice = st.sidebar.selectbox("📚 Navigate", list(PAGES.keys()))
PAGES[choice]()
