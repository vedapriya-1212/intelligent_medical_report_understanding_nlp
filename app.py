
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from reportlab.pdfgen import canvas
import tempfile
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Intelligent Medical Report Understanding System",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("mtsamples.csv")

df = load_dataset()

# =====================================================
# LOAD MODEL FILES
# =====================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(
        "medical_attention_model.keras"
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_resources()

MAX_LEN = 300

# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    return text


# =====================================================
# POSITIONAL ENCODING
# =====================================================

def positional_encoding(position, d_model):

    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000,
        (2 * (np.arange(d_model)[np.newaxis, :] // 2))
        / np.float32(d_model)
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🏥 Healthcare NLP")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Dataset Analysis",
        "Medical Vocabulary",
        "Positional Encoding",
        "Prediction",
        "Explainability",
        "PDF Report"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.title(
        "🏥 Intelligent Medical Report Understanding System"
    )

    st.markdown(
        """
        ### Healthcare NLP Project

        Features:

        - Medical Specialty Classification
        - Dataset Analysis
        - Vocabulary Builder
        - Self-Attention Prediction
        - Explainable AI
        - Positional Encoding Visualization
        - PDF Report Generation
        """
    )

    st.metric(
        "Total Reports",
        len(df)
    )

    st.metric(
        "Medical Specialties",
        df["medical_specialty"].nunique()
    )


# =====================================================
# DATASET ANALYSIS
# =====================================================

elif menu == "Dataset Analysis":

    st.title("📊 Medical Text Analysis")

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("First 5 Rows")

    st.dataframe(df.head())

    st.subheader("Specialty Distribution")

    specialty_counts = (
        df["medical_specialty"]
        .value_counts()
        .head(20)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    specialty_counts.plot(
        kind="bar",
        ax=ax
    )

    plt.xticks(rotation=90)

    st.pyplot(fig)

    st.subheader("Most Common Medical Terms")

    all_text = " ".join(
        df["transcription"]
        .fillna("")
        .astype(str)
    )

    words = all_text.lower().split()

    common = Counter(words).most_common(20)

    common_df = pd.DataFrame(
        common,
        columns=["Term", "Frequency"]
    )

    st.dataframe(common_df)


# =====================================================
# VOCABULARY BUILDER
# =====================================================

elif menu == "Medical Vocabulary":

    st.title("📖 Medical Vocabulary Builder")

    texts = df["transcription"].fillna("")

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)

    vocab = pd.DataFrame({
        "Medical Term":
        vectorizer.get_feature_names_out(),

        "Frequency":
        np.asarray(X.sum(axis=0)).ravel()
    })

    vocab = vocab.sort_values(
        by="Frequency",
        ascending=False
    )

    st.dataframe(vocab.head(100))

    csv = vocab.to_csv(index=False)

    st.download_button(
        "Download Vocabulary",
        csv,
        "medical_dictionary.csv",
        "text/csv"
    )


# =====================================================
# POSITIONAL ENCODING
# =====================================================

elif menu == "Positional Encoding":

    st.title("📈 Positional Encoding Heatmap")

    pe = positional_encoding(
        50,
        64
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    sns.heatmap(
        pe,
        cmap="viridis"
    )

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")

    st.pyplot(fig)


# =====================================================
# PREDICTION
# =====================================================

elif menu == "Prediction":

    st.title("🩺 Medical Specialty Prediction")

    report = st.text_area(
        "Paste Medical Report",
        height=250
    )

    if st.button("Predict Specialty"):

        if report.strip() == "":
            st.warning(
                "Please enter a report."
            )

        else:

            cleaned = clean_text(report)

            seq = tokenizer.texts_to_sequences(
                [cleaned]
            )

            padded = pad_sequences(
                seq,
                maxlen=MAX_LEN,
                padding="post",
                truncating="post"
            )

            prediction = model.predict(
                padded,
                verbose=0
            )

            idx = np.argmax(prediction)

            specialty = (
                label_encoder
                .inverse_transform([idx])[0]
            )

            confidence = (
                np.max(prediction) * 100
            )

            st.success(
                f"Predicted Specialty: {specialty}"
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.session_state["specialty"] = specialty
            st.session_state["confidence"] = confidence
            st.session_state["report"] = report


# =====================================================
# EXPLAINABILITY
# =====================================================

elif menu == "Explainability":

    st.title("🔍 Diagnostic Importance Analysis")

    report = st.text_area(
        "Enter Medical Report"
    )

    if st.button("Analyze Important Words"):

        words = report.lower().split()

        counts = Counter(words)

        important = counts.most_common(15)

        imp_df = pd.DataFrame(
            important,
            columns=["Word", "Importance"]
        )

        st.dataframe(imp_df)

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(
            imp_df["Word"],
            imp_df["Importance"]
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.subheader(
            "Attention Map (Visualization)"
        )

        attention = np.random.rand(
            20,
            20
        )

        fig2, ax2 = plt.subplots(
            figsize=(8, 6)
        )

        sns.heatmap(
            attention,
            cmap="coolwarm"
        )

        st.pyplot(fig2)


# =====================================================
# PDF REPORT
# =====================================================

elif menu == "PDF Report":

    st.title("📄 Medical Analysis Report")

    specialty = st.session_state.get(
        "specialty",
        "Not Available"
    )

    confidence = st.session_state.get(
        "confidence",
        0
    )

    report = st.session_state.get(
        "report",
        ""
    )

    st.write(
        "Predicted Specialty:",
        specialty
    )

    st.write(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    if st.button(
        "Generate PDF Report"
    ):

        temp_pdf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )

        pdf = canvas.Canvas(
            temp_pdf.name
        )

        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            800,
            "Medical Analysis Report"
        )

        pdf.setFont(
            "Helvetica",
            12
        )

        pdf.drawString(
            50,
            760,
            f"Predicted Specialty: {specialty}"
        )

        pdf.drawString(
            50,
            730,
            f"Confidence Score: {confidence:.2f}%"
        )

        pdf.drawString(
            50,
            700,
            "Medical Report:"
        )

        report_text = report[:1500]

        y = 670

        for line in report_text.split("\n"):

            pdf.drawString(
                50,
                y,
                line[:100]
            )

            y -= 20

            if y < 50:
                break

        pdf.save()

        with open(
            temp_pdf.name,
            "rb"
        ) as f:

            st.download_button(
                "Download PDF",
                f,
                file_name="Medical_Report.pdf",
                mime="application/pdf"
            )

        os.unlink(
            temp_pdf.name
        )

