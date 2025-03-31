import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import exifread

# Function to extract EXIF data (latitude, longitude, altitude)
def extract_exif_data(image_file):
    """
    Extract latitude, longitude, and altitude from an image's EXIF metadata.
    """
    try:
        # No usamos open() ya que `image_file` es un objeto de memoria en Streamlit
        tags = exifread.process_file(image_file)
        
        # Extract latitude, longitude, and altitude
        lat = tags.get("GPS GPSLatitude")
        lon = tags.get("GPS GPSLongitude")
        alt = tags.get("GPS GPSAltitude")
        
        # Extract reference direction (N/S, E/W)
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_ref = tags.get("GPS GPSLongitudeRef")
        
        # Convert to decimal format
        latitude = convert_gps_to_decimal(lat, lat_ref)
        longitude = convert_gps_to_decimal(lon, lon_ref)
        altitude = float(alt.values[0].num) / float(alt.values[0].den) if alt else None
        
        return {"Latitude": latitude, "Longitude": longitude, "Altitude": altitude}
    
    except Exception as e:
        return {"Latitude": None, "Longitude": None, "Altitude": None}


# Function to convert GPS coordinates from DMS to decimal
def convert_gps_to_decimal(gps_value, gps_ref):
    """
    Convert GPS coordinates from degrees, minutes, seconds (DMS) to decimal format.
    """
    if not gps_value:
        return None
    
    d = float(gps_value.values[0].num) / float(gps_value.values[0].den)
    m = float(gps_value.values[1].num) / float(gps_value.values[1].den)
    s = float(gps_value.values[2].num) / float(gps_value.values[2].den)
    
    decimal = d + (m / 60.0) + (s / 3600.0)
    
    if gps_ref and gps_ref.values[0] in ['S', 'W']:
        decimal *= -1
    
    return decimal

# Function to find overlap between two images using SIFT and visualize it with a heatmap
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
    
    # Create an overlay heatmap to highlight overlap
    overlay = img2.copy()
    alpha = 0.6  # Transparency factor
    
    for match in good_matches:
        x, y = int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1])
        cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)  # Draw heatmap points
    
    blended = cv2.addWeighted(img2, 1 - alpha, overlay, alpha, 0)  # Blend heatmap with the original image
    
    return blended, match_percentage

# Streamlit UI Title
st.title("Image Overlap Estimation with SIFT and EXIF Data")

# Upload multiple images through Streamlit file uploader
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Set threshold for redundancy filtering
threshold = st.sidebar.slider("Select Overlap Threshold (%)", min_value=0, max_value=100, value=80)

# Process images if at least two are uploaded
if len(uploaded_files) > 1:
    images = [np.array(Image.open(file)) for file in uploaded_files]
    image_paths = [file.name for file in uploaded_files]
    
    results = []
    exif_results = []
    
    for i in range(len(images)):
        exif_data = extract_exif_data(uploaded_files[i])
        exif_results.append({"Image": image_paths[i], **exif_data})
    
    exif_df = pd.DataFrame(exif_results)
    
    st.write("### Extracted EXIF Data")
    st.dataframe(exif_df)
    
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
