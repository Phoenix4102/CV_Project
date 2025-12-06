# ui.py
import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(layout="wide") # Use wide mode for better images
st.title("🏗️ SDNET2018 Crack Detection System")
st.write("Upload a concrete surface image to analyze using GeneralAD.")

# Sidebar for controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Original
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='Original Input', use_container_width=True)
    
    if st.sidebar.button("Analyze Image"):
        with st.spinner('Running GeneralAD Model...'):
            # Prepare file for API
            img_bytes = uploaded_file.getvalue()
            files = {'file': (uploaded_file.name, img_bytes, uploaded_file.type)}
            
            try:
                # Call FastAPI
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Header Result
                    pred = data['prediction']
                    if pred == "Anomaly Detected":
                        st.error(f"### ⚠️ RESULT: {pred}")
                    else:
                        st.success(f"### ✅ RESULT: {pred}")
                    
                    # Decode Images
                    heatmap = Image.open(io.BytesIO(base64.b64decode(data['heatmap_b64'])))
                    overlay = Image.open(io.BytesIO(base64.b64decode(data['overlay_b64'])))
                    bbox = Image.open(io.BytesIO(base64.b64decode(data['bbox_b64'])))
                    
                    # Layout: 3 Columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(heatmap, caption="Anomaly Heatmap (Raw Model Output)", use_container_width=True)
                        st.info("Red areas indicate high anomaly scores.")
                        
                    with col2:
                        st.image(overlay, caption="Overlay on Texture", use_container_width=True)
                        st.info("Heatmap blended with original texture.")
                        
                    with col3:
                        st.image(bbox, caption="Automated Detection", use_container_width=True)
                        st.info("Bounding box drawn based on threshold.")
                        
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is 'uvicorn app:app' running?")
