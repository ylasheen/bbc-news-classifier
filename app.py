import json
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="BBC News Classifier", page_icon="📰")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("bbc_model.keras")
    with open("vocab.json") as f:
        data = json.load(f)
    vectorizer = tf.keras.layers.TextVectorization(
        output_sequence_length=200
    )
    vectorizer.set_vocabulary(data["vectorizer_vocab"][2:])
    class_names = data["class_names"]
    return model, vectorizer, class_names

model, vectorizer, class_names = load_assets()

st.title("📰 BBC News Article Classifier")
st.write(
    "Paste any English news article text below and the model will predict "
    "which category it belongs to: **business, entertainment, politics, sport, or tech**."
)

default_text = (
    "The chancellor unveiled a new budget plan on Wednesday that includes tax cuts for "
    "small businesses and increased spending on public infrastructure. Opposition leaders "
    "criticised the proposal, arguing it would widen the deficit, while economists were "
    "divided on its likely impact on growth over the coming year. The announcement comes "
    "ahead of a general election expected later this year, with polls showing the economy "
    "remains a top concern for voters."
)
text = st.text_area("Article text (works best with a full paragraph, like a real news article)", value=default_text, height=220)

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        vec = vectorizer(tf.constant([text]))
        probs = model.predict(vec, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        st.subheader(f"Predicted category: **{class_names[pred_idx].upper()}**")
        st.write("Confidence per category:")
        for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
            st.progress(float(p), text=f"{name}: {p*100:.1f}%")

st.caption(
    "Model: Embedding + GlobalAveragePooling1D + Dense, trained on the BBC News dataset "
    "(2,225 articles, 5 categories). Train acc 99.1%, val acc 97.3%. Note: the model was "
    "trained on full-length articles, so it works best on multi-sentence text rather than "
    "single short phrases."
)
