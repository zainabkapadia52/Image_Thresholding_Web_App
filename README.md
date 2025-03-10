# Multi-Function Image Processing App

[![**Open in Streamlit**](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zainabkapadia52-image-thresholding-webapp.streamlit.app)

## Overview
This web application combines two powerful image processing tools into one interactive platform. It offers both histogram matching for photo style consistency and image thresholding using Otsu's method. In the histogram matching mode, users can upload a target style image along with one or more source images, and the app automatically transfers the tonal characteristics from the target to the source images. In the thresholding mode, users can apply automatic Otsu thresholding or adjust manually to create a binary image that effectively separates the foreground from the background. Additional features include optional Gaussian noise addition, detailed histogram visualization, and convenient download options for processed images.

## Features
- Image upload functionality
- Histogram Matching with Blending: Adjust the strength of the effect using an interactive blending slider.
- Color Image Processing: Processes images in the YCrCb color space to preserve color information.
- Otsu thresholding implementation with options of manual thresholding and gaussian noise addition.
- Interactive Visualization: View interactive histograms for the source, target, and processed images using Plotly.
- Download Options: Download the processed images directly from the interface.

## Technologies Used
- Python
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Plotly

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


