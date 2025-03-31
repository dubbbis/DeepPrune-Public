import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to compute image overlap using SIFT
def find_overlap(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    match_percentage = len(good_matches) / len(matches) * 100 if matches else 0
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, match_percentage

# Streamlit UI Title
st.title("Image Overlap Estimation")

# Upload multiple images through Streamlit file uploader
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Set threshold for redundancy filtering
threshold = st.sidebar.slider("Select Overlap Threshold (%)", min_value=0, max_value=100, value=80)

# Process images if at least three are uploaded
if len(uploaded_files) > 2:
    images = [np.array(Image.open(file)) for file in uploaded_files]
    image_paths = [file.name for file in uploaded_files]

    results = []
    images_to_keep = set(range(len(images)))  # Set of indices for images to keep
    
    for i in range(1, len(images) - 1):  # Exclude first and last images
        img_prev, img_current, img_next = images[i - 1], images[i], images[i + 1]
        prev_path, current_path, next_path = image_paths[i - 1], image_paths[i], image_paths[i + 1]
        
        # Compute overlap between (i-1) and (i) and between (i+1) and (i)
        _, match_prev = find_overlap(img_prev, img_current)
        _, match_next = find_overlap(img_next, img_current)
        _, match_skip = find_overlap(img_prev, img_next)  # Overlap between (i-1) and (i+1)
        
        # Store results
        results.append({
            "Image": current_path,
            "Overlap with Previous": match_prev,
            "Overlap with Next": match_next,
            "Overlap skipping one": match_skip,
            "Decision": "Keep" if match_skip < threshold else "Discard"
        })
        
        # If skipping one image results in high overlap, discard the middle image
        if match_skip >= threshold:
            images_to_keep.discard(i)
    
    # Display results
    for res in results:
        st.write(f"### Image: {res['Image']}")
        st.write(f"Overlap with Previous: {res['Overlap with Previous']:.2f}%")
        st.write(f"Overlap with Next: {res['Overlap with Next']:.2f}%")
        st.write(f"Overlap skipping one: {res['Overlap skipping one']:.2f}%")
        st.write(f"Decision: **{res['Decision']}**")
