import numpy as np
import tensorflow as tf
import csv, json

VOCAB_SIZE = 8000
MAX_LENGTH = 200
EMBEDDING_DIM = 16
TRAINING_SPLIT = 0.8

sentences, labels = [], []
with open("bbc-text.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentences.append(row[1])

sentences = np.array(sentences)
labels = np.array(labels)

train_size = int(len(sentences) * TRAINING_SPLIT)
idx = np.random.RandomState(42).permutation(len(sentences))
sentences, labels = sentences[idx], labels[idx]

train_sentences, val_sentences = sentences[:train_size], sentences[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LENGTH
)
vectorizer.adapt(train_sentences)

label_encoder = tf.keras.layers.StringLookup(mask_token=None, num_oov_indices=0)
label_encoder.adapt(labels)
class_names = label_encoder.get_vocabulary()
print("Classes:", class_names)

def to_ds(sents, labs):
    x = vectorizer(sents)
    y = label_encoder(labs)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(32)

train_ds = to_ds(train_sentences, train_labels)
val_ds = to_ds(val_sentences, val_labels)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=40, verbose=2, callbacks=[early_stop])

model.save("bbc_model.keras")
with open("vocab.json", "w") as f:
    json.dump({
        "vectorizer_vocab": vectorizer.get_vocabulary(),
        "class_names": class_names,
    }, f)

print("FINAL train_acc:", history.history["accuracy"][-1])
print("FINAL val_acc:", history.history["val_accuracy"][-1])
