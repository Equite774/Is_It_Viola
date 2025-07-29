import streamlit as st
import requests

st.title("Is It A Viola?")
st.subheader("Upload a clip of a {violin, viola} to classify!")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "processing_file" not in st.session_state:
    st.session_state.processing_file = False

if "response" not in st.session_state:
    st.session_state.response = None

if "results_container" not in st.session_state:
    st.session_state.results_container = st.empty()

def display_results():
    if st.session_state.response.status_code == 200:
        st.success("Prediction successful!")
        probabilities = st.session_state.response.json()["probabilities"]
        prediction = "Violin" if probabilities[0] > probabilities[1] else "Viola"
        with st.session_state.results_container:
            st.markdown(f"""
                            **Probabilities for each class:**

                            Violin: {probabilities[0]}%  
                            Viola: {probabilities[1]}%  

                            **Predicted class:** {prediction}
                            """)
    else:
        with st.session_state.results_container:
            st.error("Prediction failed.")

if uploaded_file is not None:
    if not st.session_state.processing_file:
        st.session_state.file_uploaded = True
        # Read the uploaded file
        filename = uploaded_file.name
        filestream = uploaded_file.getvalue()
        filetype = uploaded_file.type
        st.session_state.results_container.empty()
        st.write(f"Processing file: {filename}...")

        # Make a request to the FastAPI backend
        st.session_state.response = requests.post("http://localhost:8000/predict/", files={"file": (filename, filestream, filetype)})

    if st.session_state.processing_file:
        display_results()
        st.session_state.processing_file = False
        st.session_state.file_uploaded = False

    if not st.session_state.processing_file and st.session_state.file_uploaded:
        st.session_state.processing_file = True
        st.rerun()

