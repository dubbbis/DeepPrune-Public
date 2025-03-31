import streamlit as st
import cv2
import numpy as np
import pandas as pd
import exifread
# from io import BytesIO
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
    
    # Calculate deegrees, minutes, and seconds and convert to decimal format
    d = float(gps_value.values[0].num) / float(gps_value.values[0].den)
    m = float(gps_value.values[1].num) / float(gps_value.values[1].den)
    s = float(gps_value.values[2].num) / float(gps_value.values[2].den)
    decimal = d + (m / 60.0) + (s / 3600.0)
    if gps_ref and gps_ref.values[0] in ['S', 'W']:
        decimal *= -1  # Convert to negative if in the southern or western hemisphere
    return decimal

# --------------------------- GROUP IMAGES BY LOCATION --------------------------- #
def group_images_by_location(df, threshold=0.000005): # 0.000005 degrees ≈ 0.5 meters
    """Groups images based on latitude and longitude proximity within a given threshold."""
    df = df.sort_values(by=["Latitude", "Longitude"])  # Sort images by GPS coordinates
    groups = []  # List to store grouped images
    current_group = []  # Temporary list for the current group

    for i, row in df.iterrows():
        if not current_group:
            current_group.append(row["Image"])  # Start a new group with the first image
        else:
            last_image = df.loc[df["Image"] == current_group[-1]]  # Get the last image in the current group
            lat_diff = abs(row["Latitude"] - last_image["Latitude"].values[0])  # Compute latitude difference
            lon_diff = abs(row["Longitude"] - last_image["Longitude"].values[0])  # Compute longitude difference

            if lat_diff < threshold and lon_diff < threshold:
                current_group.append(row["Image"])  # If within threshold, add to the current group
            else:
                groups.append(current_group)  # If not, finalize the current group and start a new one
                current_group = [row["Image"]]

    if current_group:
        groups.append(current_group)  # Add the last group to the list

    return groups

# --------------------------- STREAMLIT INTERFACE --------------------------- #
st.title("🔍 Redundant Image Detection for Photogrammetry")

# Upload images through Streamlit UI
uploaded_files = st.file_uploader("📂 Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)

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

    # Display extracted EXIF data in a DataFrame
    exif_df = pd.DataFrame(exif_results)
    st.write("### 📍 Extracted EXIF Data")
    st.dataframe(exif_df)

    # Group images by location
    image_groups = group_images_by_location(exif_df)
    
    # Display grouped images in Streamlit
    st.write("### 📌 Image Groups by Location:")
    for idx, group in enumerate(image_groups):
        st.write(f"**Group {idx + 1}:**", group)
