import cv2
import numpy as np
import math as m

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
