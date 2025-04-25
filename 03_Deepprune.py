import streamlit as st
import cv2
import numpy as np
import pandas as pd
import exifread
from PIL import Image

# --------------------------- EXIF DATA EXTRACTION --------------------------- #
def extract_exif_data(image_file):
    try:
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
    if not gps_value:
        return None
    d = float(gps_value.values[0].num) / float(gps_value.values[0].den)
    m = float(gps_value.values[1].num) / float(gps_value.values[1].den)
    s = float(gps_value.values[2].num) / float(gps_value.values[2].den)
    decimal = d + (m / 60.0) + (s / 3600.0)
    if gps_ref and gps_ref.values[0] in ['S', 'W']:
        decimal *= -1
    return decimal

# --------------------------- IMAGE PROCESSING UTILITIES --------------------------- #
def load_and_preprocess_image(uploaded_file, size=(512, 512)):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image_np, size)
    return resized

# --------------------------- MATCH DETECTION --------------------------- #
def compute_descriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def compute_matches(des1, des2, ratio_test=0.7):
    if des1 is None or des2 is None:
        return 0
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_test * n.distance]
    return len(good_matches)

# --------------------------- STREAMLIT INTERFACE --------------------------- #
st.title("DeepPrune: Redundant Image Detection for Photogrammetry")

uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    exif_results = []
    resized_images = {}
    descriptors = {}

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        resized_image = load_and_preprocess_image(uploaded_file)
        resized_images[uploaded_file.name] = resized_image
        kp, des = compute_descriptors(resized_image)
        descriptors[uploaded_file.name] = des

        uploaded_file.seek(0)
        exif_data = extract_exif_data(uploaded_file)
        exif_results.append({"Image": uploaded_file.name, **exif_data})

    exif_df = pd.DataFrame(exif_results)
    st.write("### Extracted EXIF Data")
    st.dataframe(exif_df)

    st.sidebar.write("Parameter Settings")
    threshold_value = st.sidebar.slider("Match threshold for removing images", 5000, 30000, 10000)
    ratio_test = st.sidebar.slider("Adjust Lowe's Ratio Test", 0.5, 0.9, 0.7, 0.05)
    analyze_button = st.sidebar.button("Analyze Images")

    image_data_list = exif_df.to_dict(orient='records')

    images_to_keep = []
    image_paths = [img['Image'] for img in image_data_list if img['Latitude'] is not None]

    if analyze_button and len(image_paths) > 2:
        st.write("Analysis Results")

        images_to_keep.append(image_paths[0])

        for i in range(1, len(image_paths) - 1):
            img1_name, img2_name, img3_name = image_paths[i - 1], image_paths[i], image_paths[i + 1]

            matches_1_2 = compute_matches(descriptors[img1_name], descriptors[img2_name], ratio_test)
            matches_2_3 = compute_matches(descriptors[img2_name], descriptors[img3_name], ratio_test)
            matches_1_3 = compute_matches(descriptors[img1_name], descriptors[img3_name], ratio_test)

            img2_coincidences = matches_1_2 + matches_2_3
            discard_image_2 = img2_coincidences > threshold_value

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"{img1_name} ↔ {img2_name}: {matches_1_2} matches")
            with col2:
                st.write(f"{img2_name} ↔ {img3_name}: {matches_2_3} matches")
            with col3:
                st.write(f"{img1_name} ↔ {img3_name}: {matches_1_3} matches")

            if discard_image_2:
                st.warning(f"The image {img2_name} is redundant and will be removed.")
            else:
                st.success(f"The image {img2_name} is kept in the dataset.")
                images_to_keep.append(img2_name)

        st.success("Analysis completed!")
