import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from PIL import Image
from io import BytesIO
import random
import os

# ---------------------------------------------------
# App Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Interior Designer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Interior Designer")
st.write("Generate professional interior designs. Note: Cloud generation may take 2-4 minutes.")

# ---------------------------------------------------
# Load Model (Optimized for Cloud RAM)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    # If on Streamlit Cloud (CPU), we use a lighter model to prevent crashing
    # If you have a GPU (Local), it uses the high-quality SDXL
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.to("cuda")
    else:
        # LIGHTWEIGHT MODEL FOR CLOUD
        model_id = "runwayml/stable-diffusion-v1-5" 
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        pipe.to("cpu")
        # Critical for low-RAM environments:
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=True)
    return pipe, "cuda" if use_cuda else "cpu"

try:
    pipe, device_type = load_model()
except Exception as e:
    st.error(f"Error loading AI model: {e}")
    st.stop()

# ---------------------------------------------------
# Logic Functions
# ---------------------------------------------------
def build_prompt(room, shape, budget, color, width, length, height):
    area = width * length
    size_desc = "small" if area < 120 else "spacious"
    ceiling = "high ceilings" if height > 10 else "standard ceilings"
    
    return f"""
    (Interior shot of the INSIDE of a {room}), (looking from within the room). 
    Indoor photography, {shape} layout, {size_desc} size, {ceiling}. 
    Style: {budget} budget design, {color} color palette. 
    Professional lighting, 8k resolution, highly detailed furniture.
    """

NEGATIVE_PROMPT = "exterior, house facade, grass, trees, sky, roof, blurry, messy, distorted, watermark"

# ---------------------------------------------------
# Sidebar & UI
# ---------------------------------------------------
st.sidebar.header("Room Settings")
room_name = st.sidebar.selectbox("Room Type", ["Living Room", "Study Room", "Bedroom", "Dining Room"])
room_shape = st.sidebar.selectbox("Shape", ["Square", "Rectangle", "L-Shape"])
budget = st.sidebar.selectbox("Budget", ["lower", "middle", "higher"])

st.sidebar.subheader("Dimensions (ft)")
w = st.sidebar.slider("Width", 6, 40, 12)
l = st.sidebar.slider("Length", 6, 60, 15)
h = st.sidebar.slider("Height", 7, 18, 9)

color_theme = st.text_input("Color Theme", "modern warm wood and white")

# ---------------------------------------------------
# Execution
# ---------------------------------------------------
if st.button("Generate Design"):
    seed = random.randint(0, 10**6)
    generator = torch.Generator(device=device_type).manual_seed(seed)
    
    # Cloud (CPU) needs fewer steps to avoid timeout
    steps = 20 if device_type == "cpu" else 35

    with st.spinner(f"Generating your {room_name} on {device_type.upper()}..."):
        prompt = build_prompt(room_name, room_shape, budget, color_theme, w, l, h)
        
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        st.image(image, caption=f"Design for {room_name} (Seed: {seed})", use_container_width=True)
        
        # Download button
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button("Download Image", buf.getvalue(), "design.png", "image/png")
