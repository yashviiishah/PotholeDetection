import streamlit as st
from ultralytics import YOLO
import os
from dotenv import load_dotenv
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================
# 1️Setup
# ==============================

# Load .env from correct folder
load_dotenv(dotenv_path=r"C:\Users\Nidhi\OneDrive\Documents\Sem VII\GEN_AI\data\.env")

# Get Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Safety check — stop app if no key found
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found! Please check your .env file path or key name.")
    st.stop()

# Initialize Gemini (LangChain interface)
chat_model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)

# Load YOLO model (use raw string for Windows paths)
model_path = r"C:\Users\Nidhi\OneDrive\Documents\Sem VII\GEN_AI\best.pt"
model = YOLO(model_path)

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Pothole Detector", page_icon="🕳️", layout="centered")

st.title("🕳️ Pothole Detection & Complaint Letter Generator")
st.write("Upload a road image, detect potholes using your YOLO model, and automatically generate a complaint letter to the municipal authority.")

uploaded_file = st.file_uploader("📸 Upload a road image", type=["jpg", "jpeg", "png"])

# Default GPS values
latitude = st.number_input("📍 Latitude", value=19.113, format="%.6f")
longitude = st.number_input("📍 Longitude", value=72.869, format="%.6f")

# ==============================
# Detection + Letter Generation
# ==============================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Potholes & Generate Letter"):
        with st.spinner("Detecting potholes..."):
            # Run YOLO detection
            results = model.predict(image, conf=0.4, verbose=False)
            result = results[0]

            # Render detection image
            annotated_img = result.plot()  # Draw bounding boxes
            st.image(annotated_img, caption="Detected Potholes", use_container_width=True)

            # Extract detected classes
            detected_classes = [model.names[int(box.cls)] for box in result.boxes]

            if not any("pothole" in cls.lower() for cls in detected_classes):
                st.warning("No pothole detected in the image!")
            else:
                st.success(f"🕳️ Pothole detected! Generating complaint letter...")

                # Decide which authority based on coordinates
                if 19.0 <= latitude <= 19.2 and 72.8 <= longitude <= 72.9:
                    authority = "BMC (Mumbai)"
                elif 18.95 <= latitude <= 19.0 and 73.1 <= longitude <= 73.2:
                    authority = "NMCC (Navi Mumbai)"
                elif 19.18 <= latitude <= 19.25 and 73.0 <= longitude <= 73.05:
                    authority = "Thane Municipal Corporation"
                else:
                    authority = "Local Authority"

                # Prompt for Gemini
                prompt = f"""
                Write a formal complaint letter addressed to the {authority}.
                The pothole was detected at Latitude: {latitude}, Longitude: {longitude}.
                Mention that it causes inconvenience to commuters and should be repaired soon.
                End with a polite closing.
                """

                try:
                    # Generate response via Gemini
                    response = chat_model.invoke(prompt)
                    letter = response.content.strip()

                    st.subheader("📜 Generated Complaint Letter")
                    st.text_area("Letter", letter, height=300)

                    # Option to download
                    st.download_button(
                        label="💾 Download Letter",
                        data=letter,
                        file_name="pothole_complaint_letter.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Error generating letter: {e}")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("🚧 Built with YOLOv8 + Google Gemini + Streamlit | © 2025 Smart Road Safety Project")
