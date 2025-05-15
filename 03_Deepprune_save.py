import streamlit as st
import cv2
import numpy as np
import pandas as pd
import exifread
from PIL import Image
import time
import os

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

# --------------------------- GROUP IMAGES BY LATITUDE --------------------------- #
def group_images_by_latitude(image_data, lat_threshold=0.000005):
    grouped_images = []
    transition_images = []
    current_group = [image_data[0]]
    for i in range(1, len(image_data)):
        prev = image_data[i - 1]
        curr = image_data[i]
        lat_diff = abs(curr['Latitude'] - prev['Latitude'])
        if lat_diff <= lat_threshold:
            current_group.append(curr)
        else:
            if len(current_group) == 1:
                transition_images.append(current_group[0])
            else:
                grouped_images.append(current_group)
            current_group = [curr]
    if len(current_group) == 1:
        transition_images.append(current_group[0])
    else:
        grouped_images.append(current_group)
    return grouped_images, transition_images

# --------------------------- MATCH DETECTION AND VISUALIZATION --------------------------- #
def compute_matches_and_draw(img1, img2, ratio_test):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return 0, img2
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_test * n.distance]
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), img_matches

# --------------------------- STREAMLIT INTERFACE --------------------------- #
st.title("DeepPrune: Redundant Image Detection for Photogrammetry")

uploaded_files = st.file_uploader("Upload your images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    exif_results = []
    images = {}
    image_paths = []

    st.sidebar.write("### Parameter Settings")
    resize_width = st.sidebar.slider("Resize Width (pixels)", 320, 1280, 640, step=160)
    resize_height = st.sidebar.slider("Resize Height (pixels)", 240, 960, 480, step=120)

    threshold_value = st.sidebar.slider("Match threshold for removing images", 500, 5000, 1200)
    st.sidebar.caption("Total matches of middle image with neighbors. If too high, fewer images are removed.")

    ratio_test = st.sidebar.slider("Adjust Lowe's Ratio Test", 0.5, 0.9, 0.7, 0.05)

    overlap_threshold = st.sidebar.slider("Min overlap between image 1 and 3", 100, 1500, 500, 50)
    st.sidebar.caption("Matches between image 1 and 3. If high enough, image 2 may be removed.")

    analyze_button = st.sidebar.button("Analyze Images")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((resize_width, resize_height))
        images[uploaded_file.name] = image_resized
        image_paths.append(uploaded_file.name)
        uploaded_file.seek(0)
        exif_data = extract_exif_data(uploaded_file)
        exif_results.append({"Image": uploaded_file.name, **exif_data})

    exif_df = pd.DataFrame(exif_results)
    st.write("### Extracted EXIF Data")
    st.dataframe(exif_df)

    image_data_list = exif_df.to_dict(orient='records')
    image_groups, transition_images = group_images_by_latitude(image_data_list)

    st.write("### Image Groups by GPS Location:")
    for idx, group in enumerate(image_groups):
        with st.expander(f"Group {idx + 1} ({len(group)} images)"):
            group_images = [images[img['Image']] for img in group if img['Image'] in images]
            st.image(group_images, caption=[img['Image'] for img in group], width=150)

    if transition_images:
        st.write("### Transition Images:")
        transition_group_images = [images[img['Image']] for img in transition_images if img['Image'] in images]
        st.image(transition_group_images, caption=[img['Image'] for img in transition_images], width=150)

    images_to_keep = [image_paths[0]] if image_paths else []

    if len(image_paths) >= 3:
        t0 = time.time()
        compute_matches_and_draw(np.array(images[image_paths[0]]), np.array(images[image_paths[1]]), ratio_test)
        compute_matches_and_draw(np.array(images[image_paths[1]]), np.array(images[image_paths[2]]), ratio_test)
        t1 = time.time()
        base_time = t1 - t0
        estimated_time = (base_time / 2) * (len(image_paths) - 2)
        st.info(f"⏳ Estimated processing time: {estimated_time:.2f} seconds")

    if analyze_button and len(images) > 2:
        st.write("### Analysis Results")
        start_time = time.time()

        for i in range(1, len(image_paths) - 1):
            img1_path, img2_path, img3_path = image_paths[i - 1], image_paths[i], image_paths[i + 1]
            img1, img2, img3 = images[img1_path], images[img2_path], images[img3_path]

            matches_1_2, img_matches_1_2 = compute_matches_and_draw(np.array(img1), np.array(img2), ratio_test)
            matches_2_3, img_matches_2_3 = compute_matches_and_draw(np.array(img2), np.array(img3), ratio_test)
            matches_1_3, img_matches_1_3 = compute_matches_and_draw(np.array(img1), np.array(img3), ratio_test)

            img2_coincidences = matches_1_2 + matches_2_3
            discard_image_2 = matches_1_3 >= overlap_threshold and img2_coincidences > threshold_value

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

        end_time = time.time()
        total_time = end_time - start_time

        st.success("Analysis completed!")
        st.info(f"⏱️ Processing time: {total_time:.2f} seconds")

        st.markdown("---")
        st.subheader("Save selected images to folder")

        default_path = "./output_images"
        save_path = st.text_input("Output folder path:", value=default_path)
        save_button = st.button("💾 Save images to folder")

        if save_button:
            os.makedirs(save_path, exist_ok=True)
            for img_name in images_to_keep:
                image = images[img_name]
                image.save(os.path.join(save_path, img_name))
            st.success(f"Saved {len(images_to_keep)} images to `{save_path}`")
