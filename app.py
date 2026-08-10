import json
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="BBC News Classifier",
    page_icon=None,
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1000px;}
    h1 {font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.2rem;}
    .subtitle {color: #9aa0a6; font-size: 1.05rem; margin-bottom: 1.8rem;}
    .metric-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .section-label {
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 1px;
        color: #9aa0a6;
        margin-bottom: 0.4rem;
    }
    .result-badge {
        display: inline-block;
        background: rgba(91,141,239,0.15);
        border: 1px solid rgba(91,141,239,0.4);
        color: #7fa8f2;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("bbc_model.keras")
    with open("vocab.json") as f:
        data = json.load(f)
    vectorizer = tf.keras.layers.TextVectorization(output_sequence_length=200)
    vectorizer.set_vocabulary(data["vectorizer_vocab"][2:])
    class_names = data["class_names"]
    return model, vectorizer, class_names


model, vectorizer, class_names = load_assets()

# ---------- Header ----------
st.title("BBC News Article Classifier")
st.markdown(
    '<div class="subtitle">A text classification model that reads a news article and predicts '
    'which section it belongs to — trained on thousands of real BBC articles.</div>',
    unsafe_allow_html=True,
)

# ---------- Key stats row ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box"><div class="section-label">Categories</div>'
                '<div style="font-size:1.4rem; font-weight:600;">5</div></div>',
                unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div class="section-label">Training articles</div>'
                '<div style="font-size:1.4rem; font-weight:600;">2,225</div></div>',
                unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="section-label">Architecture</div>'
                '<div style="font-size:1.4rem; font-weight:600;">Embedding + Dense</div></div>',
                unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box"><div class="section-label">Validation accuracy</div>'
                '<div style="font-size:1.4rem; font-weight:600;">97.3%</div></div>',
                unsafe_allow_html=True)

st.write("")
st.write("")

# ---------- Explanation ----------
with st.expander("How this works", expanded=False):
    st.markdown(
        """
The model converts article text into a numeric representation using a trained word
**embedding**, then averages those word vectors into a single summary of the article
(**GlobalAveragePooling1D**) before passing it through dense layers that make the final
prediction.

**Network design:** `Embedding → GlobalAveragePooling1D → Dropout → Dense(24) → Dropout → Dense(5, softmax)`
It was trained on 2,225 real BBC articles split across five sections: business, entertainment,
politics, sport, and tech — reaching 99.1% accuracy on training data and 97.3% on unseen
validation data.

**Note:** the model was trained on full articles, so it performs best on multi-sentence text
rather than a single short phrase.
        """
    )

st.write("")

# ---------- Input ----------
st.markdown('<div class="section-label">Article text</div>', unsafe_allow_html=True)

default_text = (
    "The chancellor unveiled a new budget plan on Wednesday that includes tax cuts for "
    "small businesses and increased spending on public infrastructure. Opposition leaders "
    "criticised the proposal, arguing it would widen the deficit, while economists were "
    "divided on its likely impact on growth over the coming year. The announcement comes "
    "ahead of a general election expected later this year, with polls showing the economy "
    "remains a top concern for voters."
)

text = st.text_area(
    "Article text",
    value=default_text,
    height=220,
    label_visibility="collapsed",
)
st.caption("Paste any English news article — works best with a full paragraph, like a real news story.")

classify = st.button("Classify article", type="primary")

st.write("")

# ---------- Result ----------
if classify:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        vec = vectorizer(tf.constant([text]))
        probs = model.predict(vec, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        st.markdown('<div class="section-label">Predicted category</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="result-badge">{class_names[pred_idx].upper()}</span>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown('<div class="section-label">Confidence by category</div>', unsafe_allow_html=True)
        for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
            st.progress(float(p), text=f"{name.capitalize()} — {p*100:.1f}%")

st.write("")
st.caption(
    "Model trained on the BBC News dataset (2,225 articles, 5 categories). "
    "Train accuracy 99.1%, validation accuracy 97.3%."
)
