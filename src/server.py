import streamlit as st
import requests

st.title("Violin vs Viola Classifier")
st.subheader("Upload a clip of a violin or viola to classify!")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
if uploaded_file is not None:
    # Read the uploaded file
    filename = uploaded_file.name
    filestream = uploaded_file.getvalue()
    filetype = uploaded_file.type

    # Make a request to the FastAPI backend
    response = requests.post("http://localhost:8000/predict/", files={"file": (filename, filestream, filetype)})
    if response.status_code == 200:
        st.success("Prediction successful!")
        probabilities = response.json()["probabilities"]
        prediction = "Violin" if probabilities[0] > probabilities[1] else "Viola"
        st.write("Probabilities for each class:")
        st.write(f"Violin: {probabilities[0]}%")
        st.write(f"Viola: {probabilities[1]}%")
        st.write("Predicted class:", prediction)
    else:
        st.error("Prediction failed.")