# train.py

import pandas as pd
import numpy as np
import pickle
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = "mtsamples.csv"

MAX_WORDS = 10000
MAX_LEN = 300
EMBED_DIM = 128

EPOCHS = 10
BATCH_SIZE = 32

MODEL_PATH = "medical_attention_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


# =====================================================
# CHECK DATASET
# =====================================================

if not os.path.exists(DATASET_PATH):
    print(f"\nERROR: {DATASET_PATH} not found.")
    print("Place mtsamples.csv in the same folder as train.py")
    exit()


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully")
print("Shape :", df.shape)

print("\nColumns Found:")
print(df.columns.tolist())



# =====================================================
# SELECT REQUIRED COLUMNS
# =====================================================

required_columns = [
    "transcription",
    "medical_specialty"
]

for col in required_columns:
    if col not in df.columns:
        print(f"\nERROR: Column '{col}' not found in dataset.")
        exit()

df = df[required_columns]

df.dropna(inplace=True)

df["transcription"] = df["transcription"].astype(str)

df = df[df["transcription"].str.strip() != ""]

print("\nAfter Removing Missing Rows:")
print(df.shape)

# =====================================================
# CLEAN TEXT
# =====================================================

print("\nCleaning Medical Reports...")

df["clean_text"] = df["transcription"].apply(clean_text)

texts = df["clean_text"].tolist()

labels = df["medical_specialty"].tolist()



# =====================================================
# LABEL ENCODING
# =====================================================

print("\nEncoding Specialties...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(labels)

num_classes = len(np.unique(y))

print("Total Specialties:", num_classes)

print("\nSample Specialties:")

for cls in label_encoder.classes_[:10]:
    print(cls)


# =====================================================
# TOKENIZATION
# =====================================================

print("\nTokenizing Reports...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

vocab_size = min(
    MAX_WORDS,
    len(tokenizer.word_index) + 1
)

print("Vocabulary Size:", vocab_size)

print("Input Shape:", X.shape)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\nCreating Train/Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Train Samples:", len(X_train))
print("Test Samples :", len(X_test))


# =====================================================
# BUILD SELF ATTENTION MODEL
# =====================================================

print("\nBuilding Self-Attention Model...")


def build_attention_model():

    inputs = Input(shape=(MAX_LEN,))

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM
    )(inputs)

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32
    )(embedding, embedding)

    x = GlobalAveragePooling1D()(attention)

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = Dropout(0.3)(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = build_attention_model()

model.summary()


# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining Started...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# =====================================================
# EVALUATION
# =====================================================

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Test Accuracy:", round(accuracy * 100, 2), "%")


# =====================================================
# SAVE MODEL
# =====================================================

print("\nSaving Files...")

model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("\nSaved Successfully")

print("Model:")
print(MODEL_PATH)

print("\nTokenizer:")
print(TOKENIZER_PATH)

print("\nLabel Encoder:")
print(LABEL_ENCODER_PATH)


# =====================================================
# TRAINING COMPLETE
# =====================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

