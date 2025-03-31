import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to find overlap between two images using SIFT
def find_overlap(img1, img2):
    # Convert images to grayscale for feature extraction
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize the SIFT (Scale-Invariant Feature Transform) detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors for both images
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Define FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Initialize FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors between both images using k-nearest neighbors
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Filter good matches based on Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    # Calculate overlap percentage
    match_percentage = len(good_matches) / len(matches) * 100 if matches else 0
    
    # Draw matches between images for visualization
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, match_percentage

# Streamlit UI Title
st.title("Image Overlap Estimation with SIFT")

# Upload multiple images through Streamlit file uploader
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Set threshold for redundancy filtering
threshold = st.sidebar.slider("Select Overlap Threshold (%)", min_value=0, max_value=100, value=80)

# Process images if at least two are uploaded
if len(uploaded_files) > 1:
    images = [np.array(Image.open(file)) for file in uploaded_files]
    image_paths = [file.name for file in uploaded_files]
    
    results = []
    
    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i + 1]
        img1_path, img2_path = image_paths[i], image_paths[i + 1]
        
        # Compute overlap
        result, match_percent = find_overlap(img1, img2)
        
        # Store results
        results.append({
            "Image 1": img1_path,
            "Image 2": img2_path,
            "Overlap Percentage": match_percent
        })
        
        # Display results
        st.image(result, caption=f"Overlap Visualization: {match_percent:.2f}%", use_container_width=True)
        st.write(f"### Overlap Match Percentage: {match_percent:.2f}%")
        
        # Decision based on threshold
        if match_percent > threshold:
            st.warning(f"The second image ({img2_path}) is highly redundant and may be removed.")
        else:
            st.success(f"Both images contain distinct information and should be kept.")
