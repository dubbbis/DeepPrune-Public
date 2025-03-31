import streamlit as st
import cv2
import numpy as np
import pandas as pd
import exifread
from PIL import Image

# --------------------------- EXIF DATA EXTRACTION --------------------------- #
def extract_exif_data(image_file):
    """Extracts latitude, longitude, and altitude from an image's EXIF metadata."""
    try:
        tags = exifread.process_file(image_file)  # Read EXIF metadata from the image
        
        # Extract GPS information
        lat = tags.get("GPS GPSLatitude")
        lon = tags.get("GPS GPSLongitude")
        alt = tags.get("GPS GPSAltitude")
        
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_ref = tags.get("GPS GPSLongitudeRef")

        # Convert GPS coordinates to decimal format
        latitude = convert_gps_to_decimal(lat, lat_ref)
        longitude = convert_gps_to_decimal(lon, lon_ref)
        altitude = float(alt.values[0].num) / float(alt.values[0].den) if alt else None
        
        return {"Latitude": latitude, "Longitude": longitude, "Altitude": altitude}
    
    except Exception:
        return {"Latitude": None, "Longitude": None, "Altitude": None}

def convert_gps_to_decimal(gps_value, gps_ref):
    """Converts GPS coordinates from DMS (Degrees, Minutes, Seconds) to decimal format."""
    if not gps_value:
        return None
    d = float(gps_value.values[0].num) / float(gps_value.values[0].den)
    m = float(gps_value.values[1].num) / float(gps_value.values[1].den)
    s = float(gps_value.values[2].num) / float(gps_value.values[2].den)
    decimal = d + (m / 60.0) + (s / 3600.0)
    if gps_ref and gps_ref.values[0] in ['S', 'W']:
        decimal *= -1  # Convert to negative if in the southern or western hemisphere
    return decimal

# --------------------------- IMAGE OVERLAP DETECTION --------------------------- #
def find_overlap(img1, img2):
    """Detects overlap between two images using SIFT and FLANN algorithms."""
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply SIFT feature detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Configure FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match keypoints using KNN
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter good matches using Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Calculate overlap percentage
    match_percentage = len(good_matches) / len(matches) * 100 if matches else 0

    # Highlight matching points on the second image
    overlay = img2.copy()
    for match in good_matches:
        x, y = int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1])
        cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)
    
    # Blend the original and marked images
    blended = cv2.addWeighted(img2, 0.7, overlay, 0.3, 0)
    return blended, match_percentage

# --------------------------- STREAMLIT INTERFACE --------------------------- #
st.title("🔍 Redundant Image Detection for Photogrammetry")

# Upload images through Streamlit UI
uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    exif_results = []
    images = []
    image_paths = []
    
    for uploaded_file in uploaded_files:
        image = np.array(Image.open(uploaded_file))  # Convert uploaded image to NumPy array
        images.append(image)
        image_paths.append(uploaded_file.name)

        uploaded_file.seek(0)
        exif_data = extract_exif_data(uploaded_file)  # Extract EXIF metadata
        exif_results.append({"Image": uploaded_file.name, **exif_data})

    # Display extracted EXIF data in a table
    exif_df = pd.DataFrame(exif_results)
    st.write("### Extracted EXIF Data")
    st.dataframe(exif_df)

    # Sidebar Checkbox option to analyze image overlap
    perform_calculation = st.sidebar.checkbox("Analyze Overlap", value=False)

    if perform_calculation and len(images) > 1:
        # Define overlap threshold
        threshold = st.sidebar.slider("📏 Overlap Threshold (%)", 0, 100, 80)

        results = []
        images_to_keep = []

        for i in range(len(images) - 1):
            img1, img2 = images[i], images[i + 1]
            img1_path, img2_path = image_paths[i], image_paths[i + 1]

            result, match_percent = find_overlap(img1, img2)  # Detect overlap

            results.append({
                "Image 1": img1_path,
                "Image 2": img2_path,
                "Overlap Percentage": match_percent
            })

            # Display overlap percentage with marked keypoints
            st.image(result, caption=f"Overlap: {match_percent:.2f}%", use_container_width=True)
            st.write(f"### 📊 Match Percentage: {match_percent:.2f}%")

            # Provide recommendation based on overlap threshold
            if match_percent > threshold:
                st.warning(f"Image **{img2_path}** is redundant and can be removed.")
            else:
                st.success(f"Image **{img2_path}** is retained in the dataset.")
                images_to_keep.append(img2_path)

        # Display the final list of selected images
        st.write("### Selected Images:")
        st.write(images_to_keep)
