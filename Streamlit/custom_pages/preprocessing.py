import streamlit as st

def main():
    # Ajouter un titre stylisé sous l'image
    st.markdown(
        """
        <h1 style="color: #127395; font-size: 36px; margin-top: 20px;">
            Data preprocessing
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Section 1: Techniques d'extraction des caracteristiques avancées
    st.markdown(
        """
        <p style="font-size: 18px;">
        To enhance the analysis of X-Ray images, we used advanced feature extraction techniques such as <b>Laplacian</b>, 
        <b>Sobel</b>, <b>HOG</b>, and <b>Haralick descriptors</b> to better understand image sharpness, contours, textures, 
        and patterns. These techniques were applied in the context of Machine Learning (which is not showcased in this demo), 
        not to directly classify the images, but to provide additional insights into their characteristics and guide the analysis. 
        </p>
        <p style="font-size: 18px;">
        While Machine Learning is less powerful than Deep Learning, these methods allowed us to better explore the specific features 
        of the images, thereby improving our understanding of the input data and their potential for more advanced models.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Section 2: Laplacian Variance (Sharpness Analysis)
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Laplacian Variance (sharpness analysis)</h3>
        <p style="font-size: 18px;">
        We analyzed image sharpness using Laplacian variance, setting a threshold of <b>300</b> below which images were considered blurry.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])  # Centrer l'image en utilisant column width ratios
    with col2:
        st.image(
            "/Users/anissa/Desktop/static/Laplacian variance_Table.png",
            caption="Image sharpness and blurriness by category (without masks)",
            width=500,
    )
    st.markdown(
        """
        <p style="font-size: 18px;">
        Initially, it seemed that images with low variance lacked clarity. However, upon further inspection, we observed that these images, 
        particularly from the COVID category, were not blurry due to poor quality but rather due to pulmonary opacities characteristic 
        of COVID-19. This insight led us to decide against removing any data. 
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Section 3: Sobel Gradients (Edge Detection)
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Sobel gradients (edge detection)</h3>
        <p style="font-size: 18px;">
        The Sobel filter highlights edges and gradients. Normal images exhibit the strongest gradients, while COVID images show lower contrast, 
        likely due to ground-glass opacities.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])  # Centrer l'image en utilisant column width ratios
    with col2:
        st.image(
            "/Users/anissa/Desktop/static/SOBEL variance.png",
            caption="Histograms of Sobel gradient by category (without masks)",
            width=500,
    )

    # Section 4: HOG / Haralick 
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">HOG (shape and texture analysis) and Haralick features (statistical texture analysis)</h3>
        <p style="font-size: 18px;">
        HOG captures gradients and textures, while Haralick features analyze image textures. 
        </p>
        <p style="font-size: 18px;">
        PCA was applied to HOG features, reducing the feature dimensions from 34,596 to under 144, ensuring efficient processing while retaining essential details. 
        The 3D visualization of the principal components revealed distinct clusters between the COVID, Pneumonia, and Normal classes, although overlaps remain between COVID and Normal.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])  # Centrer l'image en utilisant column width ratios
    with col2:
        st.image(
            "/Users/anissa/Desktop/static/HOG features.png",
            caption="PCA of HOG features in 3D",
            width=450,
    )

    # Section 5: Conclusion
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">From features to modeling</h3>
        <p style="font-size: 18px;">
        By combining <b>Laplacian</b>, <b>Sobel</b>, <b>HOG</b>, and <b>Haralick features</b> , we gained deeper insights into sharpness, edges, textures, and patterns. 
        The radiographic raw images and the features were integrated into a structured CSV file, ready for use in the modeling phase.
        </p>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
