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

st.markdown(
    """
    <style>
    .section-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    .result-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 1.1rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("bbc_model.keras")

    with open("vocab.json") as f:
        data = json.load(f)

    vectorizer = tf.keras.layers.TextVectorization(
        output_sequence_length=200
    )

    vectorizer.set_vocabulary(
        data["vectorizer_vocab"][2:]
    )

    class_names = data["class_names"]

    return model, vectorizer, class_names


model, vectorizer, class_names = load_assets()


# ---------- Header ----------

st.title("BBC News Article Classifier")

st.markdown(
    "A text classification model that reads a news article and predicts "
    "which section it belongs to — trained on thousands of real BBC articles."
)


# ---------- Key stats row ----------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        **Categories**  
        5
        """
    )

with col2:
    st.markdown(
        """
        **Training articles**  
        2,225
        """
    )

with col3:
    st.markdown(
        """
        **Architecture**  
        Embedding + Dense
        """
    )

with col4:
    st.markdown(
        """
        **Validation accuracy**  
        97.3%
        """
    )


st.write("")
st.write("")


# ---------- Explanation ----------

with st.expander("How this works", expanded=False):
    st.markdown(
        """
        The model converts article text into a numeric representation using a trained
        word **embedding**, then averages those word vectors into a single summary of
        the article (**GlobalAveragePooling1D**) before passing it through dense layers
        that make the final prediction.

        **Network design:**
        `Embedding → GlobalAveragePooling1D → Dropout → Dense(24) → Dropout → Dense(5, softmax)`

        It was trained on 2,225 real BBC articles split across five sections:
        business, entertainment, politics, sport, and tech — reaching 99.1% accuracy
        on training data and 97.3% on unseen validation data.

        **Note:** the model was trained on full articles, so it performs best on
        multi-sentence text rather than a single short phrase.
        """
    )


st.write("")


# ---------- Input ----------

st.markdown(
    "**Article text**"
)

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

st.caption(
    "Paste any English news article — works best with a full paragraph, like a real news story."
)


# ---------- Short Text Warning ----------

word_count = len(text.split())

if 0 < word_count < 30:
    st.warning(
        "⚠️ النص قصير جدًا، جرّب فقرة أطول عشان نتيجة أدق."
    )


# ---------- Classify Button ----------

classify = st.button(
    "Classify article",
    type="primary"
)

st.write("")


# ---------- Result ----------

if classify:

    # Empty text
    if not text.strip():

        st.warning(
            "Please enter some text."
        )

    # Text is too short
    elif word_count < 30:

        st.warning(
            "⚠️ النص قصير جدًا، جرّب فقرة أطول عشان نتيجة أدق."
        )

    # Valid article
    else:

        vec = vectorizer(
            tf.constant([text])
        )

        probs = model.predict(
            vec,
            verbose=0
        )[0]

        pred_idx = int(
            np.argmax(probs)
        )


        # ---------- Prediction ----------

        st.markdown(
            '<div class="section-label">Predicted category</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<span class="result-badge">{class_names[pred_idx].upper()}</span>',
            unsafe_allow_html=True,
        )


        st.write("")


        # ---------- Confidence ----------

        st.markdown(
            '<div class="section-label">Confidence by category</div>',
            unsafe_allow_html=True,
        )

        for name, p in sorted(
            zip(class_names, probs),
            key=lambda x: -x[1]
        ):

            st.progress(
                float(p),
                text=f"{name.capitalize()} — {p * 100:.1f}%"
            )


# ---------- Footer ----------

st.write("")

st.caption(
    "Model trained on the BBC News dataset (2,225 articles, 5 categories). "
    "Train accuracy 99.1%, validation accuracy 97.3%."
)
