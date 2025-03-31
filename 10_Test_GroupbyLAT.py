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
def group_images_by_latitude(image_data):
    """Groups images based only on latitude threshold."""
    lat_threshold = 0.000005  # Adjusted latitude threshold for grouping within a column
    
    grouped_images = []  # Store the groups
    current_group = [image_data[0]]  # Start with the first image
    
    for i in range(1, len(image_data)):
        prev = image_data[i - 1]
        curr = image_data[i]
        
        lat_diff = abs(curr['Latitude'] - prev['Latitude'])
        
        if lat_diff <= lat_threshold:
            current_group.append(curr)  # Add to current group if within threshold
        else:
            grouped_images.append(current_group)  # Save the finished group
            current_group = [curr]  # Start a new group
    
    grouped_images.append(current_group)  # Add the last group
    return grouped_images

# --------------------------- STREAMLIT INTERFACE --------------------------- #
st.title("Redundant Image Detection for Photogrammetry")

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

    # Display extracted EXIF data in a DataFrame
    exif_df = pd.DataFrame(exif_results)
    st.write("### Extracted EXIF Data")
    st.dataframe(exif_df)

    # Convert DataFrame to list of dictionaries for processing
    image_data_list = exif_df.to_dict(orient='records')
    
    # Group images by latitude using the calibrated threshold
    image_groups = group_images_by_latitude(image_data_list)
    
    # Display grouped images in Streamlit
    st.write("### Image Groups by Latitude:")
    for idx, group in enumerate(image_groups):
        st.write(f"Group {idx + 1}:", [img['Image'] for img in group])
