import streamlit as st

st.title("Camera Input Test")

img = st.camera_input("Take a picture")

if img is not None:
    st.image(img, caption="Captured Image", use_column_width=True)
