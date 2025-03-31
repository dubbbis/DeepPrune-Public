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

# --------------------------- GROUP IMAGES BY LATITUDE --------------------------- #
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

# --------------------------- MATCH DETECTION AND VISUALIZATION --------------------------- #
def compute_matches_and_draw(img1, img2, ratio_test):
    """Detects matching points between two images, draws them, and returns the image with matches."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # If no descriptors are found, return 0 overlap
    if des1 is None or des2 is None:
        return 0, img2 # If not descriptors are found, return 0 matches and the original image

    # FLANN parameters and matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test to find good matches (matches with a distance ratio less than 0.7)
    good_matches = [m for m, n in matches if m.distance < ratio_test * n.distance]
    

    # Draw matching points on the second image
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return len(good_matches), img_matches  # Return number of matches and the image with drawn points

# --------------------------- STREAMLIT INTERFACE --------------------------- #

# Set Streamlit title
st.title("DeepPrune: Redundant Image Detection for Photogrammetry")

# Upload images through Streamlit UI
uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    exif_results = []
    images = {}
    image_paths = []
    # Lists to store image data


    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file)  # Open image using PIL
        images[uploaded_file.name] = image  # Store the image in a dictionary
        image_paths.append(uploaded_file.name) # Save image filename

        uploaded_file.seek(0)

        # Extract EXIF data from the image
        exif_data = extract_exif_data(uploaded_file)  # Extract EXIF metadata
        exif_results.append({"Image": uploaded_file.name, **exif_data}) # Store the data

    # Display extracted EXIF data in a DataFrame
    exif_df = pd.DataFrame(exif_results)
    st.write("### Extracted EXIF Data") # Section title
    st.dataframe(exif_df) # Show EXIF data in a table

    # Sidebar for parameter settings

    st.sidebar.write("Parameter Settings") # Sidebar title
    threshold_value = st.sidebar.slider("Match threshold for removing images", 5000, 30000, 10000) # Slider to set the threshold for image redundancy
    ratio_test = st.sidebar.slider("Adjust Lowe's Ratio Test", 0.5, 0.9, 0.7, 0.05)  # Slider to set Lowe's Ratio Test parameter
    st.sidebar.write("") # Add extra space before the button
    analyze_button = st.sidebar.button("Analyze Images") # Button to start processing

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

    # # Compute overlap button
    # if st.sidebar.button("Compute Overlap"):
    #     st.write("### Image Overlap Analysis")
    #     for group in image_groups:
    #         for i in range(len(group) - 1):
    #             img1_name = group[i]['Image']
    #             img2_name = group[i + 1]['Image']
    #             overlap = extract_exif_data(images[img1_name], images[img2_name])
    #             st.write(f"Overlap between {img1_name} and {img2_name}: {overlap:.2f}%")


    # Display images and results
    images_to_keep = [image_paths[0]]

    if analyze_button and len(images) > 2:
        st.write("Analysis Results")  
        # Section title for results
        
        results = []

        for i in range(1, len(image_paths) - 1):
            img1_path, img2_path, img3_path = image_paths[i - 1], image_paths[i], image_paths[i + 1]
            img1, img2, img3 = images[img1_path], images[img2_path], images[img3_path]  # Acceder con nombres de archivo

            matches_1_2, img_matches_1_2 = compute_matches_and_draw(np.array(img1), np.array(img2), ratio_test)
            matches_2_3, img_matches_2_3 = compute_matches_and_draw(np.array(img2), np.array(img3), ratio_test)
            matches_1_3, img_matches_1_3 = compute_matches_and_draw(np.array(img1), np.array(img3), ratio_test)

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

    # images_to_keep.append(image_paths[-1])

    # st.write("Selected Images:")
    # st.write(images_to_keep)