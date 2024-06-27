import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def detect_text(image):
    # Convert the image to OpenCV format
    image_cv = np.array(image)
    image_cv = image_cv[:, :, ::-1].copy()

    # Preprocess the image
    processed_image = preprocess_image(image_cv)

    # Use Tesseract to extract text
    custom_config = r'--oem 3 --psm 6'  # Using psm 6 for block of text
    text = pytesseract.image_to_string(processed_image, config=custom_config)

    return text.strip()

def main():
    st.title("Text Detection and OCR")

    # File upload and processing
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect text and perform OCR
        ocr_text = detect_text(image)

        # Display the OCR text
        st.header("OCR Result")
        st.subheader(f"Detected text: {ocr_text}")

if __name__ == "__main__":
    main()
