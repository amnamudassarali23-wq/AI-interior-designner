import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from PIL import Image
from io import BytesIO
import random

# ---------------------------------------------------
# App Configuration
# ---------------------------------------------------
st.set_page_config(page_title="AI Interior Designer", page_icon="🏠", layout="wide")

st.title("🏠 AI Interior Designer")
st.write("Professional interior visualization. (Note: Initial load takes ~2 mins)")

# ---------------------------------------------------
# Model Loader (Memory Optimized for Cloud)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        # High-quality model if you have a GPU (Local)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.to("cuda")
    else:
        # LIGHTWEIGHT MODEL for Streamlit Cloud (CPU) to prevent crashes
        model_id = "runwayml/stable-diffusion-v1-5" 
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        pipe.to("cpu")
        # Optimization for low RAM
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=True)
    return pipe, "cuda" if use_cuda else "cpu"

try:
    pipe, device_type = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}. Ensure requirements.txt is present on GitHub.")
    st.stop()

# ---------------------------------------------------
# Sidebar & UI
# ---------------------------------------------------
st.sidebar.header("Room Configuration")
room_name = st.sidebar.selectbox("Room Type", ["Living Room", "Study Room", "Bedroom", "Dining Room"])
budget = st.sidebar.selectbox("Budget Style", ["Lower (Simple)", "Middle (Modern)", "Higher (Luxury)"])

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
    
    # Fewer steps on Cloud/CPU to avoid timing out
    steps = 20 if device_type == "cpu" else 35

    with st.spinner(f"Designing your {room_name} on {device_type.upper()}..."):
        # Stronger prompt engineering to force interior view
        prompt = f"(Interior shot of the INSIDE of a {room_name}), looking from within the room. "\
                 f"Dimensions: {w}x{l}ft, {h}ft ceiling. Style: {budget}. Theme: {color_theme}. "\
                 f"Professional lighting, 8k resolution, highly detailed."
        
        neg_prompt = "exterior, house facade, grass, trees, sky, roof, blurry, messy, distorted"

        image = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        st.image(image, caption=f"Design for {room_name} (Seed: {seed})", use_container_width=True)
        
        # Download logic
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button("Download Image", buf.getvalue(), "interior_design.png", "image/png")
