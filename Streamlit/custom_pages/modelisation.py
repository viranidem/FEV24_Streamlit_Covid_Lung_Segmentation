import streamlit as st
import os
import glob
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2  # Assure-toi d'avoir installé OpenCV : pip install opencv-python


# Définir la fonction Grad-CAM
def get_gradcam_and_prediction(model, image, last_conv_layer_name, pred_index=None):
    """
    Applique Grad-CAM sur une image pour visualiser les activations
    et retourne également la classe prédite.
    """
    # Créer un modèle intermédiaire pour capturer les activations et prédictions
    grad_model = tf.keras.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(image)
        conv_outputs, predictions = grad_model(image)

        # Classe prédite
        predicted_class = tf.argmax(predictions[0]).numpy()

        # Par défaut, utiliser la classe prédite si pred_index n'est pas spécifié
        if pred_index is None:
            pred_index = predicted_class

        class_channel = predictions[:, pred_index]

    # Calcul des gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Grad-CAM : Moyenne des gradients sur les dimensions spatiales
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Appliquer les gradients au résultat de la couche de convolution
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Moyenne des canaux pour obtenir la carte d'activation
    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalisation de la heatmap
    heatmap = np.maximum(heatmap, 0)  # Ne garder que les valeurs positives
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalisation

    return heatmap, predicted_class
def main():
    st.markdown(
        """
        <h1 style="color: #127395; font-size: 36px; margin-top: 20px;">
            Making X-rays speak : EfficientNet and Grad-CAM
        </h1>
        """,
        unsafe_allow_html=True
    )

# Titre 1: How deep learning helps with lung X-ray images
    st.markdown(
    """
    <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">How deep learning helps with lung X-ray images</h3>
    <p style="font-size: 18px;">
    Deep learning offers a powerful ability to analyze lung X-rays by capturing subtle features essential for accurate diagnosis. 
    Unlike traditional machine learning approaches, convolutional neural networks (CNNs) can process complex images while adapting to 
    the specificities of medical data.
    </p>
    <p style="font-size: 18px;">
    During this project, several deep learning models were tested, including DenseNet121, VGG16, and custom CNN architectures. 
    <b>EfficientNetB0</b> was ultimately chosen for its superior performance on our dataset :
    <ul style="font-size: 18px;">
        <li>Overall <b>accuracy</b>: <b>95%</b></li>
        <li><b>Recall</b> for COVID cases: <b>94%</b></li>
    </ul>
    """,
    unsafe_allow_html=True
)

# Titre 2: Dive into model performance: Test the images
    st.markdown(
        """
        <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Dive into model performance: Test the images</h3>
        <p style="font-size: 18px;">
        The performance achieved with EfficientNetB0 translates into precise and robust classifications. 
        However, to better understand the results and validate the model's reliability, it’s essential to test various images under real-world conditions.
        </p>
        """,
        unsafe_allow_html=True
    )


    # Fonction pour récupérer les chemins des images
    def paths_to_image(working_dir):
        covid_image_path = os.path.join(base_path, 'COVID')
        normal_image_path = os.path.join(base_path, 'Normal')
        pneumonia_image_path = os.path.join(base_path, 'Pneumonia')

        all_covid_image_paths = glob.glob(os.path.join(covid_image_path, '*.jpg')) + glob.glob(os.path.join(covid_image_path, '*.png'))
        all_normal_image_paths = glob.glob(os.path.join(normal_image_path, '*.jpg')) + glob.glob(os.path.join(normal_image_path, '*.png'))
        all_pneumonia_image_paths = glob.glob(os.path.join(pneumonia_image_path, '*.jpg')) + glob.glob(os.path.join(pneumonia_image_path, '*.png'))

        all_image_paths = all_covid_image_paths + all_normal_image_paths + all_pneumonia_image_paths

        return all_image_paths, all_covid_image_paths, all_normal_image_paths, all_pneumonia_image_paths

    # Charger le modèle
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("static/model2bis_enc.keras")

    model = load_model()

    # Fonction de prétraitement
    def preprocess_image(image):
        img = np.array(image.convert("L"))
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_rgb = np.stack([img] * 3, axis=-1)
        img_rgb = cv2.resize(img_rgb, (256, 256))
        img_rgb = np.expand_dims(img_rgb, axis=0)
        return img_rgb

    # Spécifie le chemin de ton répertoire local
    base_path = 'static/Data_lung_segmentation_Drive'

    # Charger les chemins des images
    all_image_paths, all_covid_image_paths, all_normal_image_paths, all_pneumonia_image_paths = paths_to_image(base_path)

    # Sélectionner une catégorie
    category = st.selectbox("**1. Choose a category from Normal, COVID, Pneumonia** :", ["COVID", "Normal", "Pneumonia"])

    # Charger les images correspondantes
    if category == "COVID":
        image_paths = all_covid_image_paths
    elif category == "Normal":
        image_paths = all_normal_image_paths
    else:
        image_paths = all_pneumonia_image_paths

    # Vérifier si des images sont disponibles
    if len(image_paths) > 0:
        st.markdown(
        f"<p style='color: grey;'>Number of images in the category {category}: {len(image_paths)}</p>",
         unsafe_allow_html=True
)
        # Sélectionner une image
        selected_image = st.selectbox("**2. Select an image, then click 'Analyse the image' to test the predictions** :", image_paths)
        image = Image.open(selected_image)

        # Afficher l'image avec une largeur fixe
        st.image(image, caption=f"Selected image : {os.path.basename(selected_image)}", width=300)

        # Effectuer une prédiction sur l'image
        if st.button("Analyse the image"):
            processed_image = preprocess_image(image)
            preds = model.predict(processed_image)

            # Afficher les résultats
            class_names = ["COVID", "Normal", "Pneumonia"]
            predicted_class = np.argmax(preds, axis=-1)
            st.write(f"Classe prédite : **{class_names[predicted_class[0]]}**")

            st.write("Probabilités associées :")
            for i, prob in enumerate(preds[0]):
                st.write(f"{class_names[i]} : {prob:.2%}")
    else:
        st.warning("Aucune image disponible dans cette catégorie.")


    # Titre 3: Grad-CAM: See what the model sees
    st.markdown(
    """
    <h3 style="color: #127395; font-size: 22px; margin-top: 20px;">Grad-CAM: See what the model sees</h3>
    <p style="font-size: 18px;">
        Grad-CAM (Gradient-weighted Class Activation Mapping) is a powerful tool for visualizing the areas of X-rays that influenced the model’s predictions. 
        This method enhances transparency by highlighting the lung regions activated by the model:
    </p>
    <ul style="font-size: 18px;">
        <li><b>Bright colors</b> (red, yellow): These indicate the regions where the model detected features it considered most important for classification.</li>
        <li><b>Darker colors</b> (blue, green): These areas are less relevant to the model's decision.</li>
    </ul>
    <p style="font-size: 18px;">
        By focusing on the bright areas, you can understand which parts of the X-ray influenced the model’s output the most.
    </p>
    <p style="font-size: 18px;">
        In this final step, to better understand why an image was classified into a specific category, use Grad-CAM to visualize the areas of X-rays that 
        had the most influence on the model's predictions.
    </p>
    """,
    unsafe_allow_html=True
)

    # Bouton pour générer Grad-CAM
    if st.button("Generate Grad-CAM"):
        # Prétraiter l'image sélectionnée
        processed_image = preprocess_image(image)
        image_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)
        
        # Appliquer Grad-CAM
        last_conv_layer_name = 'top_conv'  # Remplacez par le nom de la dernière couche conv de votre modèle
        heatmap, predicted_class = get_gradcam_and_prediction(model, image_tensor, last_conv_layer_name)

        # Map des classes
        class_names = ["COVID", "Normal", "Pneumonia"]
        predicted_class_name = class_names[predicted_class]

        # Redimensionner la heatmap à la taille de l'image d'origine
        heatmap_view = cv2.resize(heatmap, (256, 256))
        heatmap_view = np.uint8(255 * heatmap_view)  # Convertir la heatmap en uint8
        heatmap_view = cv2.applyColorMap(heatmap_view, cv2.COLORMAP_JET)  # Appliquer un colormap

        # Superposition de la heatmap sur l'image d'origine
        superimposed_image = cv2.addWeighted(processed_image[0].astype('uint8'), 0.7, heatmap_view, 0.3, 0)

        # Afficher les résultats côte à côte
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(processed_image[0] / 255.0, caption="Original Image", width=300)
        with col2:
            st.image(heatmap_view, caption="Grad-CAM Heatmap", width=300)
        with col3:
            st.image(superimposed_image, caption=f"Superimposed Image\nPredicted Class: {predicted_class_name}", width=300)

    # Explications sous les images
    st.markdown(
        """
        In general, the distribution of highlighted areas in a Grad-CAM reflects typical patterns seen in these conditions:

        - **COVID cases**: Lesions often appear in the lower lung regions and near the lung periphery, showing ground-glass opacities or other inflammation.
        - **Viral pneumonia**: Activations tend to be more diffuse, often affecting both lower lobes and the outer edges of the lungs.
        - **Bacterial pneumonia**: Highlights may concentrate on a specific lobe or area, often in the upper or middle parts of the lung, due to localized consolidation.

        These patterns are typical but not definitive, as lung conditions vary between individuals. Grad-CAM helps provide transparency by showing how the model reaches its conclusions.
        """,
        unsafe_allow_html=True
)
    # Phrase explicative pour motif de non inclusion du dataset
    st.markdown(
    """
    **Note:** Due to deployment constraints and the need for fast performance, we opted to use a random selection of approximately 500 images from each category for this demonstration. The complete dataset, which contains more than 33,000 images, could not be included.
    """,
    unsafe_allow_html=True
)
    
