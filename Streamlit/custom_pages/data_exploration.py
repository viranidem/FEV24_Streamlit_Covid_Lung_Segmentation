import streamlit as st

def main():
    # Ajoutée un titre stylisé 
    st.markdown(
        """
        <h1 style="color: #127395; font-size: 36px; margin-top: 20px;">
            Data exploration
        </h1>
        """,
        unsafe_allow_html=True
    )

    # INTRODUCTION
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Dataset selection : From challenges to solutions</h3>
        """,
        unsafe_allow_html=True,
    )

    # Contenu pour la section
    st.markdown(
        """
        <p style="font-size: 18px;">
            Initially, our project relied on a first dataset which, despite its diverse range of images, 
            exhibited a significant class imbalance and various biases, complicating the creation of a balanced 
            and unbiased model.
        </p>
        <p style="font-size: 18px;">
            To overcome these limitations, we conducted extensive online research and ultimately selected an 
            updated dataset: <b>COVID-QU-Ex</b> (<a href="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu" 
            target="_blank" style="color: #127395;">source</a>). Created by the same team responsible for the original 
            dataset, COVID-QU-Ex addresses the class imbalance issue by incorporating additional data to achieve a 
            more balanced distribution of categories.
        </p>
        <p style="font-size: 18px;">
            As a result, we successfully reduced some of the initial biases and significantly improved the reliability 
            of our model for detecting COVID-19 and other pulmonary infections.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Section Pie chart and histogramme
    
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Data distribution in the COVID-QU-Ex dataset</h3>
        <p style="font-size: 18px;">The COVID-QU-Ex dataset contains X-ray images (.png) classified into three categories: <b>Normal</b>, <b>COVID</b>, and <b>Pneumonia</b>. Each category includes subfolders for images and their corresponding masks, enabling advanced deep learning techniques. All images and masks have a uniform size of <b>256 x 256 pixels.</p>
        <p style="font-size: 18px;">The balanced distribution of images across categories is illustrated below through a <b>Histogram</b> and a <b>Pie Chart</b>. These visualizations confirm the even representation of classes, a critical improvement over the original dataset.</p>
        """,
        unsafe_allow_html=True,
    )

    # Afficher  l'histogramme sur la colonne du milieu 
    col1, col2, col3 = st.columns([1, 2, 1])  # Center image using column width ratios
    with col2:
        st.image(
            "/Users/anissa/Desktop/static/Number of images per category.png",
            caption="Number of images per category",
            width=500,
        )

    # Afficher le pie chart sur la colonne du milieu
    col1, col2, col3 = st.columns([1, 2, 1])  # Centrer l'image en utilisant column width ratios
    with col2:
        st.image(
            "/Users/anissa/Desktop/static/Percentage of images per category.png",
            caption="Percentage of images per category",
            width=500,
        )

    # Section titre
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Visualizing the dataset solutions</h3>
        <p style="font-size: 18px;">To better understand the dataset, we present sample visualizations for each category: <b>Normal</b>, <b>COVID</b>, and <b>Pneumonia</b>. These visualizations include :</p>
            <li><b>Image Only:</b> The original X-ray.</li>
            <li><b>Mask Only:</b> The mask highlighting the region of interest.</li>
            <li><b>Masked Image:</b> The X-ray overlaid with the corresponding mask.</li>
        """,
        unsafe_allow_html=True,
    )

    # Ligne 1: Normal
    st.markdown("#### Normal")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("/Users/anissa/Desktop/static/Normal_Image only.png", caption="Image Only", width=300)
    with col2:
        st.image("/Users/anissa/Desktop/static/Normal_Mask only.png", caption="Mask Only", width=300)
    with col3:
        st.image("/Users/anissa/Desktop/static/Normal_Masked image.png", caption="Masked Image", width=300)

    # Ligne 2: COVID-19
    st.markdown("#### COVID")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("/Users/anissa/Desktop/static/COVID_Image only.png", caption="Image Only", width=300)
    with col2:
        st.image("/Users/anissa/Desktop/static/COVID_Mask only.png", caption="Mask Only", width=300)
    with col3:
        st.image("/Users/anissa/Desktop/static/COVID_Masked image.png", caption="Masked Image", width=300)

    # Ligne 3: Pneumonia
    st.markdown("#### Pneumonia")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("/Users/anissa/Desktop/static/Pneumonia_Image only.png", caption="Image Only", width=300)
    with col2:
        st.image("/Users/anissa/Desktop/static/Pneumonia_Mask only.png", caption="Mask Only", width=300)
    with col3:
        st.image("/Users/anissa/Desktop/static/Pneumonia_Masked image.png", caption="Masked Image", width=300)
