import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="AI Interior Designer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Interior Designer")
st.write("AI-based interior design generator")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Room Configuration")

room_name = st.sidebar.selectbox(
    "Room Type",
    ["Bed Room", "Study Room", "Living Room", "Drawing Room", "Dining Room", "TV Lounge"]
)

room_shape = st.sidebar.selectbox(
    "Room Shape",
    ["Square", "Rectangle", "L-Shape"]
)

budget = st.sidebar.selectbox(
    "Budget Level",
    ["lower", "middle", "higher"]
)

st.sidebar.subheader("Room Dimensions (feet)")
width = st.sidebar.number_input("Width", 6.0, 40.0, 12.0)
length = st.sidebar.number_input("Length", 6.0, 60.0, 15.0)
height = st.sidebar.number_input("Height", 7.0, 18.0, 9.0)

color_theme = st.text_input(
    "Color Theme",
    "modern neutral tones with wooden accents"
)

# --------------------------------------------------
# Prompt Display
# --------------------------------------------------
prompt = f"""
Interior design of a {room_name}
Room shape: {room_shape}
Budget: {budget}
Dimensions: {width}x{length}x{height} feet
Color theme: {color_theme}
"""

st.subheader("Generated Design Prompt")
st.code(prompt)

# --------------------------------------------------
# Generate Button (Safe Mode)
# --------------------------------------------------
if st.button("Generate Interior Design"):
    st.warning("⚠ AI image generation requires GPU and runs locally.")
    st.info("This Streamlit Cloud version demonstrates the project logic.")

    st.image(
        "https://images.unsplash.com/photo-1600585154340-be6161a56a0c",
        caption="Sample Interior Design Output (Demo)",
        use_container_width=True
    )
