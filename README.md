# Image Thresholding Web App

[![**Open in Streamlit**](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zainabkapadia52-image-thresholding-webapp.streamlit.app)

## Overview
This web application implements Otsu's thresholding method for image binarization. Image thresholding is a fundamental image processing technique that separates an image's foreground from its background, creating a binary image. Otsu's method is an automatic thresholding algorithm that calculates the optimal threshold value to separate an image into foreground and background. It works by minimizing the intra-class variance between the two classes of pixels.
Tis webapp provides an interactive interface for users to upload images, apply thresholding, and visualize the results.

## Features
- Image upload functionality
- Otsu thresholding implementation
- Manual thresholding option
- Gaussian noise addition (optional)
- Histogram visualization
- Download options for processed images

## Technologies Used
- Python
- Streamlit
- OpenCV
- NumPy
- Matplotlib

## Installation
1. Clone the repository: git clone https://github.com/zainabkapadia52/Image_Thresholding_Web_App.git
2. Navigate to the project directory: cd Image_Thresholding_Web_App
3. Install the required dependencies: pip install -r requirements.txt

## Usage
1. Run the Streamlit app:
2. Open your web browser and go to `http://localhost:8501`
3. Upload an image and experiment with different thresholding options

## Contributing
Contributions to improve the app are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.


