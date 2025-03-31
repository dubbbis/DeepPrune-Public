# Import the necessary libraries
import streamlit as st
import cv2
import numpy as np
from PIL import Image # Lybrairies to work with images (Pillow)



# Define a function to find overlap between two images using SIFT
def find_overlap(img1, img2):
    # Convert images to grayscale for feature extraction
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
      # Convierte ambas imágenes a escala de grises, ya que SIFT funciona mejor con imágenes en este formato.
      # Convert both images to grayscale as SIFT works better with images in this format.

    # Initialize the SIFT (Scale-Invariant Feature Transform) detector
    sift = cv2.SIFT_create() # Crea un objeto SIFT para detectar keypoints y calcular descriptores en las imágenes.
                             # Create a SIFT object to detect keypoints and compute descriptors in the images.

    
    # Detect keypoints and compute descriptors for both images # 
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
        # Encuentra los puntos clave y genera los descriptores de cada imagen.
        # Find keypoints and generate descriptors for each image.

#Configuración del Matcher FLANN

    # Define FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
       
    # Initialize FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors between both images using k-nearest neighbors
    matches = flann.knnMatch(des1, des2, k=2)
    # Inicializa el matcher basado en FLANN. Utiliza FLANN para encontrar similitudes entre los puntos clave de ambas imágenes.
    # Initialize FLANN-based matcher. Use FLANN to find similarities between the keypoints of both images.
    
    # Filter good matches based on Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance] # Aplica el test de razón de Lowe, descartando coincidencias poco confiables. 

    
    # Calculate overlap percentage
    match_percentage = len(good_matches) / len(matches) * 100 if matches else 0 # Calcula el porcentaje de coincidencia basado en los puntos clave detectados.

    
    # Draw matches between images for visualization 
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
    # Dibuja líneas entre los puntos clave coincidentes en ambas imágenes.

    
    return img_matches, match_percentage # Devuelve la imagen con las coincidencias visualizadas y el porcentaje de superposición.
                                         # Returns the image with matches visualized and the overlap percentage.   


# Streamlit UI Title
st.title("Image Overlap Estimation with SIFT")

# Upload images through Streamlit file uploader
uploaded_file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])
# Permite a los usuarios subir dos imágenes a través de la interfaz de Streamlit. 
# Allows users to upload two images through the Streamlit interface.


# Set threshold for redundancy filtering
threshold = st.sidebar.slider("Select Overlap Threshold (%)", min_value=0, max_value=100, value=80)

# Process images if both are uploaded
if uploaded_file1 and uploaded_file2:

##  Code to transform images into CV compatible format
    # Convert uploaded images to NumPy arrays for OpenCV processing
    img1 = np.array(Image.open(uploaded_file1)) # Open the image format PIL.image
    img2 = np.array(Image.open(uploaded_file2))
    # Se asegura de que ambas imágenes sean subidas antes de continuar.

    # Convert images from RGB to BGR for OpenCV compatibility
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    # Convierte las imágenes subidas a formato NumPy y las transforma de RGB a BGR para su compatibilidad con OpenCV.

## Code to find overlap between images
    # Compute overlap and visualize results
    result, match_percent = find_overlap(img1, img2)
    
    # Display results in Streamlit interface
    st.image(result, caption=f"Overlap Visualization - Match: {match_percent:.2f}%", use_container_width=True)
    st.write(f"### Overlap Match Percentage: {match_percent:.2f}%")
    
    # Decision based on threshold
    if match_percent > threshold:
        st.warning("The second image is highly redundant and may be removed from the dataset.")
    else:
        st.success("Both images contain distinct information and should be kept.")
