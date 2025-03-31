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
    
    # Calculate degrees, minutes, and seconds and convert to decimal format
    d = float(gps_value.values[0].num) / float(gps_value.values[0].den)
    m = float(gps_value.values[1].num) / float(gps_value.values[1].den)
    s = float(gps_value.values[2].num) / float(gps_value.values[2].den)
    decimal = d + (m / 60.0) + (s / 3600.0)
    if gps_ref and gps_ref.values[0] in ['S', 'W']:
        decimal *= -1  # Convert to negative if in the southern or western hemisphere
    return decimal

# --------------------------- GROUP IMAGES BY LATITUDE ONLY --------------------------- #
def group_images_by_latitude(image_data, lat_threshold=0.000005):
    """Groups images based only on latitude threshold and identifies transition images."""
    grouped_images = []  # Store the groups
    transition_images = []  # Store transition images
    
    current_group = [image_data[0]]  # Start with the first image
    
    for i in range(1, len(image_data)):
        prev = image_data[i - 1]
        curr = image_data[i]
        
        lat_diff = abs(curr['Latitude'] - prev['Latitude'])
        
        if lat_diff <= lat_threshold:
            current_group.append(curr)  # Add to current group if within threshold
        else:
            # If the previous group contains only one image, classify it as a transition image
            if len(current_group) == 1:
                transition_images.append(current_group[0])
            else:
                grouped_images.append(current_group)  # Save the finished group
            
            current_group = [curr]  # Start a new group
    
    # Add the last group or classify it as transition
    if len(current_group) == 1:
        transition_images.append(current_group[0])
    else:
        grouped_images.append(current_group)
    
    return grouped_images, transition_images

# --------------------------- IMAGE OVERLAP DETECTION --------------------------- #
def find_overlap(img1, img2):
    """Detects overlap between two images using SIFT and FLANN algorithms."""
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0  # No descriptors found, no overlap

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    match_percentage = len(good_matches) / len(matches) * 100 if matches else 0
    
    return match_percentage

# --------------------------- STREAMLIT INTERFACE --------------------------- #
st.title("Redundant Image Detection for Photogrammetry")

# Upload images through Streamlit UI
uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    exif_results = []
    images = {}
    image_paths = []
    
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)  # Open image using PIL
        images[uploaded_file.name] = image  # Store the image in a dictionary
        image_paths.append(uploaded_file.name)

        uploaded_file.seek(0)
        exif_data = extract_exif_data(uploaded_file)  # Extract EXIF metadata
        exif_results.append({"Image": uploaded_file.name, **exif_data})

    # Display extracted EXIF data in a DataFrame
    exif_df = pd.DataFrame(exif_results)
    st.write("### Extracted EXIF Data")
    st.dataframe(exif_df)

    # Convert DataFrame to list of dictionaries for processing
    image_data_list = exif_df.to_dict(orient='records')
    
    # Group images by latitude and detect transition images
    image_groups, transition_images = group_images_by_latitude(image_data_list)
    
    # Display grouped images in Streamlit
    st.write("### Image Groups by GPS Location:")
    for idx, group in enumerate(image_groups):
        with st.expander(f"Group {idx + 1} ({len(group)} images)"):
            group_images = [images[img['Image']] for img in group if img['Image'] in images]
            st.image(group_images, caption=[img['Image'] for img in group], width=150)
    
    # Display transition images separately
    if transition_images:
        st.write("### Transition Images:")
        transition_group_images = [images[img['Image']] for img in transition_images if img['Image'] in images]
        st.image(transition_group_images, caption=[img['Image'] for img in transition_images], width=150)

    # Compute overlap button
    if st.sidebar.button("Compute Overlap"):
        st.write("### Image Overlap Analysis")
        for group in image_groups:
            for i in range(len(group) - 1):
                img1_name = group[i]['Image']
                img2_name = group[i + 1]['Image']
                overlap = find_overlap(images[img1_name], images[img2_name])
                st.write(f"Overlap between {img1_name} and {img2_name}: {overlap:.2f}%")
