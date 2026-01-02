import streamlit as st
from openai import OpenAI

# Initialize client (Assuming OpenAI)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🏡 AI Dream Room Designer")

# Sidebar for Requirements
with st.sidebar:
    st.header("Room Specifications")
    room_type = st.selectbox("Room Type", ["Living Room", "Bedroom", "Office", "Kitchen"])
    shape = st.selectbox("Room Shape", ["Rectangular", "Square", "L-Shaped"])
    dim_w = st.number_input("Width (ft)", value=12)
    dim_l = st.number_input("Length (ft)", value=15)
    
    st.header("Aesthetics")
    primary_color = st.color_picker("Primary Color", "#F0F0F0")
    contrast_color = st.color_picker("Contrast Color", "#2F4F4F")
    style = st.text_input("Style (e.g., Mid-century Modern, Minimalist)", "Modern")

if st.button("Generate Design"):
    # Combine inputs into a detailed prompt
    prompt = f"A professional interior design photo of a {shape} {room_type}, " \
             f"dimensions approximately {dim_w}x{dim_l} feet. " \
             f"The color palette is based on {primary_color} with {contrast_color} accents. " \
             f"Style: {style}. Highly detailed, 8k resolution, photorealistic."
    
    with st.spinner("Designing your space..."):
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        st.image(image_url, caption=f"Your {style} {room_type}")
