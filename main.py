import cv2
import pytesseract
import numpy as np
import imutils
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

def detect_license_plate(image):
    # Convert the image to OpenCV format
    image_cv = np.array(image)
    image_cv = image_cv[:, :, ::-1].copy()

    # Preprocess the image
    processed_image = preprocess_image(image_cv)

    # Use Canny edge detection
    edged = cv2.Canny(processed_image, 30, 200)

    # Find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Initialize license plate contour and ROI
    license_plate_contour = None
    roi = None

    # Loop through contours to find the license plate
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # Filter by aspect ratio and size
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 2 <= aspect_ratio <= 5:  # Typical aspect ratio for license plates
                license_plate_contour = approx
                roi = image_cv[y:y + h, x:x + w]
                break

    # If ROI is found, apply OCR
    if roi is not None:
        # Preprocess ROI for better OCR results
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Use Tesseract to extract text
        custom_config = r'--oem 3 --psm 8'
        text = pytesseract.image_to_string(roi_thresh, config=custom_config)

        return text.strip()
    else:
        return "License plate not detected"

def main():
    st.title("License Plate Detection and OCR")

    # File upload and processing
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect license plate and perform OCR
        ocr_text = detect_license_plate(image)

        # Display the OCR text
        st.header("OCR Result")
        st.subheader(f"Detected license plate number: {ocr_text}")

if __name__ == "__main__":
    main()