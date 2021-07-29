import streamlit as st
import numpy as np
import cv2
from classifier import Classifier


@st.cache(allow_output_mutation=True)
def load_model():
    return Classifier()


model = load_model()


def main():
    st.title("Gender Detection")

    uploaded_file = st.file_uploader('Upload an image file')

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cv2image = cv2.imdecode(file_bytes, 1)

        st.image(
            cv2image, caption='Uploaded Image',
            use_column_width=True,
            channels="BGR"
        )
        st.write("")
        st.write("Predicting ...")

        labeled_image = model._process(cv2image)

        st.image(
            labeled_image, caption='Prediction',
            use_column_width=True,
            channels="BGR"
        )


if __name__ == '__main__':
    main()
