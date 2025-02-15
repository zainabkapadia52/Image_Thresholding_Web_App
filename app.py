# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
from utils import *

st.set_page_config(page_title="Enhanced Image Thresholding App", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
}
.stTextInput > div > div > input {
    background-color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("placeholder.jpg", width=100)  # Replace with your logo
    st.title("Controls")
    apply_blur = st.checkbox("Apply Gaussian Blur")
    if apply_blur:
        kernel_size = st.slider("Kernel Size", 1, 31, 5, step=2)
        sigma = st.slider("Sigma", 0.1, 10.0, 1.0, step=0.1)

# Main content
st.title("🖼️ Enhanced Image Thresholding App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    
    if apply_blur:
        img = apply_gaussian_blur(img, kernel_size, sigma)
        with col2:
            st.image(img, caption="Blurred Image", use_container_width=True)

    method = st.radio("Select Thresholding Method", ["Otsu", "Manual"])

    if method == "Otsu":
        t = find_threshold_within(img)
        thresholded_img = otsu_img_within(img)
        st.write(f"Otsu Threshold: {t}")
    else:
        t = st.slider("Select Threshold", 0, 255, 128)
        thresholded_img = manual_threshold(img, t)

    col3, col4 = st.columns(2)
    with col3:
        st.image(thresholded_img, caption="Thresholded Image", use_container_width=True)

    with col4:
        hist = np.bincount(img.reshape(-1), minlength=256)
        fig = px.bar(x=range(256), y=hist, labels={'x': 'Pixel Intensity', 'y': 'Frequency'})
        fig.add_vline(x=t, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Add Noise"):
        if st.button("Add Gaussian Noise"):
            noisy_img = gaussian_noise(img)
            st.image(noisy_img, caption="Noisy Image", use_container_width=True)
            noisy_thresholded = otsu_img_within(noisy_img) if method == "Otsu" else manual_threshold(noisy_img, t)
            st.image(noisy_thresholded, caption="Thresholded Noisy Image", use_container_width=True)

    st.header("Download Processed Images")
    col5, col6, col7 = st.columns(3)
    with col5:
        buf = BytesIO()
        plt.imsave(buf, img, cmap='gray', format='png')
        st.download_button(label="Download Original Image", data=buf.getvalue(), file_name="original_image.png", mime="image/png")
    with col6:
        buf = BytesIO()
        plt.imsave(buf, thresholded_img, cmap='gray', format='png')
        st.download_button(label="Download Thresholded Image", data=buf.getvalue(), file_name="thresholded_image.png", mime="image/png")
    with col7:
        buf = BytesIO()
        fig.write_image(buf, format='png')
        st.download_button(label="Download Histogram", data=buf.getvalue(), file_name="histogram.png", mime="image/png")
