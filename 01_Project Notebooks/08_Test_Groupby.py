import streamlit as st  
import cv2  
import numpy as np  
import pandas as pd  
import exifread  
from PIL import Image  
from sklearn.cluster import DBSCAN  # Clustering algorithm to group images based on GPS location

def extract_exif_data(image_file):
    """Extract latitude, longitude, and altitude from EXIF metadata."""
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
    """Convert GPS coordinates from DMS to decimal format."""
    if not gps_value:
        return None
    d = float(gps_value.values[0].num) / float(gps_value.values[0].den)
    m = float(gps_value.values[1].num) / float(gps_value.values[1].den)
    s = float(gps_value.values[2].num) / float(gps_value.values[2].den)
    decimal = d + (m / 60.0) + (s / 3600.0)

    if gps_ref and gps_ref.values[0] in ['S', 'W']:
        decimal *= -1
    return decimal

def group_images_by_position(df, eps=0.0001, min_samples=2):
    """Group images based on their GPS position using DBSCAN."""
    coords = df[["Latitude", "Longitude"]].dropna().values  # Extract valid coordinates

    if len(coords) == 0:
        return None  # If no valid coordinates, return nothing

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine").fit(np.radians(coords))  
    # eps: Defines max distance (in radians) to be considered the same group (~10 meters)

    df["Group"] = clustering.labels_  # Assign a group to each image in the DataFrame

    return df

st.title("Redundant Image Detection with Position-Based Grouping")

uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)  
# Allows the user to upload multiple images

if uploaded_files:
    exif_results, images, image_paths = [], [], []

    for uploaded_file in uploaded_files:
        image = np.array(Image.open(uploaded_file))  
        # Convert the image to a NumPy array
        images.append(image)
        image_paths.append(uploaded_file.name)  
        # Save the image filename
        uploaded_file.seek(0)  
        # Reset file pointer

        exif_data = extract_exif_data(uploaded_file)
        # Extract EXIF metadata
        exif_results.append({"Image": uploaded_file.name, **exif_data})  
        # Store EXIF data

    df_exif = pd.DataFrame(exif_results)
    st.write("Extracted EXIF Data")
    # Display extracted EXIF data
    st.dataframe(df_exif)

    st.sidebar.write("Parameter Configuration")
    # Sidebar title

    threshold_value = st.sidebar.slider("Match threshold for removing images", 5000, 30000, 10000)
    # Slider to set the threshold for image redundancy

    ratio_test = st.sidebar.slider("Adjust Lowe's Ratio Test", 0.5, 0.9, 0.7, 0.05)
    # Slider to set Lowe's Ratio Test parameter

    eps_value = st.sidebar.slider("Maximum distance to group images (latitude/longitude degrees)", 0.00001, 0.001, 0.0001)
    # Slider to adjust grouping sensitivity

    st.sidebar.write("")  # Extra space before the button

    analyze_button = st.sidebar.button("Analyze Images")  
    # Button to start image processing

    if analyze_button and len(images) > 2:
        st.write("Analysis Results")  
        # Section title for results

        df_grouped = group_images_by_position(df_exif, eps=eps_value)
        # Apply image grouping based on position

        if df_grouped is not None:
            st.write("Images grouped by position:")
            # Show grouped images
            st.dataframe(df_grouped[["Image", "Latitude", "Longitude", "Group"]])  
