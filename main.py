import cv2
import imutils
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from utils import create_scanned_document, detect_contours, detect_edges, save_as_pdf

st.set_page_config(page_title="Document Scanner", page_icon=":pencil:", layout="wide")

st.title("Document Scanner")
st.write("Upload an image and we'll scan it for you!")

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Original Image", use_column_width=True)

    image_pil = Image.fromarray(image)

    cropping_choice = st.checkbox("Manually Crop the image")

    if cropping_choice:
        st.write("Crop the region you want to scan.")
        cropped_image_pil = st_cropper(
            image_pil, realtime_update=True, aspect_ratio=None
        )

        # Convert cropped PIL image back to OpenCV format
        cropped_image = np.array(cropped_image_pil)

        h, w, _ = cropped_image.shape
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

        # Apply perspective transform to get the scanned document
        scanned_document = create_scanned_document(cropped_image, pts, 1.0)
    else:
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = imutils.resize(image, height=500)
        edges = detect_edges(image)
        contours = detect_contours(edges)
        scanned_document = create_scanned_document(orig, contours, ratio)

    st.image(scanned_document, caption="Scanned Document", use_column_width=True)

    pdf_data = save_as_pdf(scanned_document)
    if st.download_button(
        label="Download Scanned Document",
        data=pdf_data,
        file_name="scanned_document.pdf",
        mime="application/pdf",
    ):
        st.success("PDF downloaded successfully!")
else:
    st.write("Please upload an image file.")
