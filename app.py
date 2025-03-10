# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
from utils import *


st.set_page_config(page_title="Multi-Function Image Processing App", layout="wide")

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

# Sidebar: Select application mode
with st.sidebar:
    st.title("Controls")
    app_mode = st.radio("Select Application", ("Histogram Matching", "Image Thresholding"))
    
    if app_mode == "Histogram Matching":
        blend = st.slider("Blending Factor (0.0 = original, 1.0 = full effect)", 0.0, 1.0, 1.0, step=0.1)
        st.markdown("---")
        st.write("Upload a target style image and one or more source images. The target image determines the tonal distribution that will be transferred to the source images via histogram matching.")
    else:
        st.image("placeholder.jpg", width=100)  # Replace with your own image/logo if desired
        apply_blur = st.checkbox("Apply Gaussian Blur")
        if apply_blur:
            kernel_size = st.slider("Kernel Size", 1, 31, 5, step=2)
            sigma = st.slider("Sigma", 0.1, 10.0, 1.0, step=0.1)

# Main Content

if app_mode == "Histogram Matching":
    st.title("📸 Photo Style Consistency App - Histogram Matching")
    st.write("This app applies histogram matching to transfer the style (contrast, brightness, and tonal distribution) from a target image to one or more source images.")
    
    target_file = st.file_uploader("Upload Target Style Image", type=["jpg", "jpeg", "png"])
    source_files = st.file_uploader("Upload Source Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if target_file is not None and source_files:
        # Process the target image
        target_bytes = np.asarray(bytearray(target_file.read()), dtype=np.uint8)
        target_img = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)
        
        target_ycrcb = cv2.cvtColor(target_img, cv2.COLOR_BGR2YCrCb)
        target_Y = target_ycrcb[:, :, 0]
        eq_target, tgt_cdf, _ = equalized_histogram(target_Y)
        
        st.subheader("Target Style Image")
        st.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), caption="Target Image", use_container_width=True)
        st.plotly_chart(plot_histogram(target_Y, "Target Histogram"), use_container_width=True, key="target_hist_main")
        
        # Process each source image
        for idx, source_file in enumerate(source_files):
            source_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
            source_img = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
            
            result_img, source_Y, _, matched_Y = process_color_image(source_img, target_img, blend=blend)
            
            st.markdown("---")
            st.subheader(f"Processed Image: {source_file.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB), caption="Original Source Image", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
            
            st.write("### Histograms")
            col_hist1, col_hist2, col_hist3 = st.columns(3)
            with col_hist1:
                st.plotly_chart(plot_histogram(source_Y, "Source Histogram"), use_container_width=True, key=f"source_hist_{idx}")
            with col_hist2:
                st.plotly_chart(plot_histogram(target_Y, "Target Histogram"), use_container_width=True, key=f"target_hist_{idx}")
            with col_hist3:
                st.plotly_chart(plot_histogram(matched_Y, "Matched Histogram"), use_container_width=True, key=f"matched_hist_{idx}")
            
            # Download button
            buf = BytesIO()
            plt.imsave(buf, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), format='png')
            st.download_button(
                label="Download Processed Image",
                data=buf.getvalue(),
                file_name=f"processed_{source_file.name}",
                mime="image/png",
                key=f"download_{idx}"
            )

elif app_mode == "Image Thresholding":
    st.title("🖼️ Enhanced Image Thresholding App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        
        if 'apply_blur' in locals() and apply_blur:
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
            hist = np.bincount(img.ravel(), minlength=256)
            fig = px.bar(x=list(range(256)), y=hist, labels={'x': 'Pixel Intensity', 'y': 'Frequency'})
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