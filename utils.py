import cv2
import numpy as np
import math as m
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px

# Helper Functions for Histogram Matching

def equalized_histogram(source_img):
    height, width = source_img.shape
    hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            intensity = source_img[i, j]
            hist[intensity] += 1
    total_pixels = np.sum(hist)
    hist_norm = hist / total_pixels
    cdf = np.cumsum(hist_norm)
    mapped_bins = np.round(cdf * 255).astype(np.uint8)
    eq_img = mapped_bins[source_img]
    eq_hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            intensity = eq_img[i, j]
            eq_hist[intensity] += 1
    eq_cdf = np.cumsum(eq_hist / total_pixels)
    return eq_img, eq_cdf, eq_hist

def find_value_target(val, target_cdf):
    min_diff = float('inf')
    chosen_index = 0
    for i in range(256):
        diff = abs(target_cdf[i] - val)
        if diff < min_diff:
            min_diff = diff
            chosen_index = i
    return chosen_index

def eq_match_histogram(eq_source_img, src_cdf, eq_target_img, tgt_cdf, blend=1.0):
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = find_value_target(src_cdf[i], tgt_cdf)
    height, width = eq_source_img.shape
    matched_img = np.zeros_like(eq_source_img)
    for i in range(height):
        for j in range(width):
            new_val = mapping[eq_source_img[i, j]]
            matched_img[i, j] = np.uint8(blend * new_val + (1 - blend) * eq_source_img[i, j])
    return matched_img

def process_color_image(source_img, target_img, blend=1.0):
    # Convert both images from BGR to YCrCb
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    target_ycrcb = cv2.cvtColor(target_img, cv2.COLOR_BGR2YCrCb)
    
    # Use the Y (luminance) channel for histogram matching
    source_Y = source_ycrcb[:, :, 0]
    target_Y = target_ycrcb[:, :, 0]
    
    eq_source, src_cdf, _ = equalized_histogram(source_Y)
    eq_target, tgt_cdf, _ = equalized_histogram(target_Y)
    matched_Y = eq_match_histogram(eq_source, src_cdf, eq_target, tgt_cdf, blend=blend)
    
    # Replace Y channel in the source image with the matched Y channel
    source_ycrcb[:, :, 0] = matched_Y
    result_img = cv2.cvtColor(source_ycrcb, cv2.COLOR_YCrCb2BGR)
    return result_img, source_Y, target_Y, matched_Y

def plot_histogram(image, title='Histogram'):
    hist = np.bincount(image.ravel(), minlength=256)
    fig = px.bar(x=list(range(256)), y=hist, labels={'x': 'Pixel Intensity', 'y': 'Frequency'})
    fig.update_layout(title=title)
    return fig

# Helper Functions for Image Thresholding

def gaussian_noise(img):
    gaussian_noise = np.random.randint(0, 128, size=len(np.asarray(img).reshape(-1)), dtype=np.uint8).reshape(img.shape)
    noisy_img = cv2.add(img, gaussian_noise)
    return noisy_img

def variance_within(t, hist):
    try:
        w1 = sum(hist[i] for i in range(0, t))
        mu1 = sum(i * hist[i] for i in range(0, t)) / sum(hist[i] for i in range(0, t))
        var1 = sum(hist[x] * ((x - mu1)**2) for x in range(0, t)) / sum(hist[i] for i in range(0, t))

        w2 = sum(hist[i] for i in range(t, len(hist)))
        mu2 = sum(i * hist[i] for i in range(t, len(hist))) / sum(hist[i] for i in range(t, len(hist)))
        var2 = sum(hist[x] * ((x - mu2)**2) for x in range(t, len(hist))) / sum(hist[i] for i in range(t, len(hist)))
        var = w1 * var1 + w2 * var2
    except:
        var = float(m.inf)
    return var

def find_threshold_within(img):
    pix = np.asarray(img).reshape(-1)
    hist = np.bincount(pix, minlength=256)
    thresh = 0
    t = 1
    min_var = float(m.inf)
    while t <= 255:
        Vw = variance_within(t, hist)
        if min_var > Vw:
            min_var = Vw
            thresh = t
        t = t + 1
    return thresh

def otsu_img_within(img):
    new_img = np.zeros_like(img)
    threshold = find_threshold_within(img)
    new_img[img >= threshold] = 255
    return new_img

def manual_threshold(image, threshold):
    return (image > threshold).astype(np.uint8) * 255

def apply_gaussian_blur(image, kernel_size, sigma):
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
