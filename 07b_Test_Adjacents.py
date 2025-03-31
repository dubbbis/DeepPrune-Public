import streamlit as st
import cv2
import numpy as np
import pandas as pd
import exifread
from PIL import Image

# --------------------------- EXIF DATA EXTRACTION --------------------------- #
def extract_exif_data(image_file):
    """Extracts latitude, longitude, and altitude from an image's EXIF metadata using exifread."""
    try:
        image_file.seek(0)  
        tags = exifread.process_file(image_file)

        lat = tags.get("GPS GPSLatitude")
        lon = tags.get("GPS GPSLongitude")
        alt = tags.get("GPS GPSAltitude")

        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_ref = tags.get("GPS GPSLongitudeRef")

        latitude = convert_gps_to_decimal(lat, lat_ref)
        longitude = convert_gps_to_decimal(lon, lon_ref)
        altitude = float(alt.values[0].num) / float(alt.values[0].den) if alt else None

        return {"Latitude": latitude, "Longitude": longitude, "Altitude": altitude}
    
    except Exception:
        return {"Latitude": None, "Longitude": None, "Altitude": None}

def convert_gps_to_decimal(gps_value, gps_ref):
    """Converts GPS coordinates from DMS (degrees, minutes, seconds) to decimal format."""
    if not gps_value:
        return None
    d = float(gps_value.values[0].num) / float(gps_value.values[0].den)
    m = float(gps_value.values[1].num) / float(gps_value.values[1].den)
    s = float(gps_value.values[2].num) / float(gps_value.values[2].den)
    decimal = d + (m / 60.0) + (s / 3600.0)

    if gps_ref and gps_ref.values[0] in ['S', 'W']:
        decimal *= -1
    return decimal

# --------------------------- MATCH DETECTION AND VISUALIZATION --------------------------- #
def compute_matches_and_draw(img1, img2, ratio_test):
    """Detects matching points between two images, draws them, and returns the image with matches."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0, img2  # If no descriptors are found, return 0 matches and the original image

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < ratio_test * n.distance]

    # Draw matching points on the second image
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return len(good_matches), img_matches  # Return number of matches and the image with drawn points

# --------------------------- USER INTERFACE --------------------------- #

st.title("DeepPrune: Redundant Image Detection for Photogrammetry")  
# Streamlit title for the app

uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)  
# Allows user to upload multiple images

if uploaded_files:
    exif_results, images, image_paths = [], [], []  
    # Lists to store image data

    for uploaded_file in uploaded_files:
        image = np.array(Image.open(uploaded_file))  
        # Convert image to NumPy array
        images.append(image)
        image_paths.append(uploaded_file.name)  
        # Save image filename
        uploaded_file.seek(0)  
        # Reset file pointer

        exif_data = extract_exif_data(uploaded_file)  
        # Extract EXIF metadata
        exif_results.append({"Image": uploaded_file.name, **exif_data})  
        # Store data

    st.write("Extracted EXIF Data")  
    # Display extracted EXIF data
    st.dataframe(pd.DataFrame(exif_results))  
    # Show EXIF data in a table

    st.sidebar.write("Parameter Settings")  
    # Sidebar title

    threshold_value = st.sidebar.slider("Match threshold for removing images", 5000, 30000, 10000)  
    # Slider to set the threshold for image redundancy

    ratio_test = st.sidebar.slider("Adjust Lowe's Ratio Test", 0.5, 0.9, 0.7, 0.05)  
    # Slider to set Lowe's Ratio Test parameter

    st.sidebar.write("")  
    # Add extra space before the button

    analyze_button = st.sidebar.button("Analyze Images")  
    # Button to start processing

    if analyze_button and len(images) > 2:
        st.write("Analysis Results")  
        # Section title for results
        
        results = []
        images_to_keep = [image_paths[0]]

        for i in range(1, len(images) - 1):
            img1, img2, img3 = images[i - 1], images[i], images[i + 1]
            img1_path, img2_path, img3_path = image_paths[i - 1], image_paths[i], image_paths[i + 1]

            matches_1_2, img_matches_1_2 = compute_matches_and_draw(img1, img2, ratio_test)
            matches_2_3, img_matches_2_3 = compute_matches_and_draw(img2, img3, ratio_test)
            matches_1_3, img_matches_1_3 = compute_matches_and_draw(img1, img3, ratio_test)

            img2_coincidences = matches_1_2 + matches_2_3
            discard_image_2 = img2_coincidences > threshold_value and (i + 1) not in [3, 5]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(cv2.resize(img_matches_1_2, (300, 300)), caption=f"{img1_path} ↔ {img2_path}", use_container_width=True)
                st.write(f"Matches: {matches_1_2}")

            with col2:
                st.image(cv2.resize(img_matches_2_3, (300, 300)), caption=f"{img2_path} ↔ {img3_path}", use_container_width=True)
                st.write(f"Matches: {matches_2_3}")

            with col3:
                st.image(cv2.resize(img_matches_1_3, (300, 300)), caption=f"{img1_path} ↔ {img3_path}", use_container_width=True)
                st.write(f"Matches: {matches_1_3}")

            if discard_image_2:
                st.warning(f"The image {img2_path} is redundant and will be removed.")
            else:
                st.success(f"The image {img2_path} is kept in the dataset.")
                images_to_keep.append(img2_path)

        images_to_keep.append(image_paths[-1])

        st.write("Selected Images:")
        st.write(images_to_keep)
