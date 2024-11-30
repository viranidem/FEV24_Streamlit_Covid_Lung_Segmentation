import streamlit as st

def main():
    # Ajouter un titre sous l'image
    st.markdown(
        """
        <h1 style="color: #127395; font-size: 36px; margin-top: 20px;">
            About our project
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Ajouter du contenu sous le titre
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">A glimpse into our study</h3>
        <p style="font-size: 18px;">
            This demo is part of a broader study dedicated to improving diagnostic methods for COVID-19 detection using chest X-rays. 
            The larger project explores both machine learning and deep learning techniques, addressing challenges such as class imbalance, 
            data preprocessing, and interpretability in medical imaging.
        </p>
        <p style="font-size: 18px;">
            In this demo, we focus specifically on deep learning to classify radiographic images effectively.
        </p>

        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">The COVID-19 context</h3>
        <p style="font-size: 18px;">
            The COVID-19 pandemic exposed critical gaps in healthcare systems, particularly in the timely diagnosis of infectious diseases. 
            This study aims to address these challenges by leveraging advanced techniques tailored to medical imaging.
        </p>

        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Objectives</h3>
        <ul style="font-size: 18px;">
            <li><b>Classify radiographic images</b> into three categories: Normal, COVID, and Pneumonia.</li>
            <li>Perform <b>data exploration</b> and <b>preprocessing</b> steps.</li>
            <li>Use <b>Grad-CAM</b> to <b>interpret model predictions</b> and highlight areas of interest in the images.</li>
        </ul>

        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Model highlights</h3>
        <ul style="font-size: 18px;">
            <li>Implements a <b>pretrained EfficientNetB0 model</b>, fine-tuned for high-accuracy classification tasks.</li>
            <li>Model weights are stored in <b>model_weights.weights.h5</b> for reproducibility and efficient deployment.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )


