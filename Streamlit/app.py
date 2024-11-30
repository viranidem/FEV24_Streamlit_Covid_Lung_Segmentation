import os
import streamlit as st
from PIL import Image  # Pour charger les images locales

# Configuration globale
st.set_page_config(
    page_title="Deep learning with EfficientNet and Grad-CAM for COVID-19 diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour personnaliser la sidebar
custom_css = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #127395;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
        padding-bottom: 10px;
    }
    [data-testid="stSidebar"] h1 {
        font-weight: bold !important;
        color: white !important;
        font-size: 22px !important;
        margin-bottom: 15px !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
        font-weight: normal !important;
    }

     [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .stRadio > label {
        margin-bottom: 0 !important;
    }
    .sidebar-content {
        flex-grow: 1;
    
    }
    /* Supprimer l'espace entre le texte et le menu radio */
    .stRadio > div {
        margin-top: -15px !important;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Charger le logo
logo_path = "/Users/anissa/Desktop/Projet DS_COVID 19 CXR_2024/Logo_DS.png"

# Contenu de la barre latérale
with st.sidebar:
    st.markdown("<h1>Radiographic Image Classifier</h1>", unsafe_allow_html=True)
    
    st.write("Navigate through different sections :")
    menu_selection = st.radio(
        "",
        options=[
            "Get started", 
            "Behind the project", 
            "Explore the data", 
            "Process the data", 
            "Model insights", 
        ],
        label_visibility="collapsed"  # Ceci cache le label vide
    )

# Fond 
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #FFFFFF, #FFEDD5);
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Charger et afficher les pages
if menu_selection == "Get started":
    from custom_pages.home import main  
    main()  
elif menu_selection == "Behind the project":
    from custom_pages.about_our_project import main  # Charger la page About
    main()
elif menu_selection == "Explore the data":
    from custom_pages.data_exploration import main  # Charger la page Data Exploration
    main()
elif menu_selection == "Process the data":
    from custom_pages.preprocessing import main  # Charger la page Pre-processing
    main()
elif menu_selection == "Model insights":
    from custom_pages.modelisation import main  # Charger la page Modélisation
    main()

