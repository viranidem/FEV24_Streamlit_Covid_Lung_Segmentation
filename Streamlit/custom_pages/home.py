import streamlit as st

def main():
    # Main content area with columns for centering
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios to center content

    # Place the image in the middle column
    with col2:
        image_path = "/Users/anissa/Desktop/static/Image_Demo_Streamlit.png"  
        try:
            st.image(image_path, use_container_width=False, width=650)  # Set image size
        except FileNotFoundError:
            st.warning("Image not found. Please check the path.")

    # Add a stylized title under the image

    st.markdown(
    """
    <h1 style="color: #127395; font-size: 36px; margin-top: 20px;">
        Welcome to the COVID-19 diagnostic tool !
    </h1>
    """,
    unsafe_allow_html=True
)

    #  Add content below the title
    st.markdown(
    """
    <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">What this tool offers</h3>
    <p>
        This application leverages chest X-ray images and cutting-edge deep learning techniques to provide efficient COVID-19 detection :
    </p>
    <ul>
        <li><b>Interactive exploration:</b> Navigate through different sections to explore the data, preprocessing steps, and results. See how the EfficientNetB0 model identifies COVID-19 cases and understand its predictions using Grad-CAM visualizations.</li>
        <li><b>Simplified presentation:</b> Built for everyone—whether you're a data enthusiast or just curious—this interface offers a clear view of the methods and outcomes.</li>
    </ul>
    
    <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Where it all started</h3>
    <p>
        This project was developed during a part-time training program for data scientists at 
        <a href="https://datascientest.com" target="_blank">Datascientest</a>. Learn more about their programs 
        <a href="https://datascientest.com/formations-data" target="_blank">here</a>.
    </p>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()

