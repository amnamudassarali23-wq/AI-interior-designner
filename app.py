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
st.write("Professional interior visualization powered by AI.")

# ---------------------------------------------------
# Model Loader (Memory Optimized)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    # Detect if we have a GPU (Local) or just CPU (Cloud)
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        # High-quality model for GPU
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.to("cuda")
    else:
        # Lighter model for Cloud/CPU to stay under 1GB RAM limit
        model_id = "runwayml/stable-diffusion-v1-5" 
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        pipe.to("cpu")
        # Magic command to save memory:
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=True)
    return pipe, "cuda" if use_cuda else "cpu"

try:
    pipe, device = load_model()
except Exception as e:
    st.error(f"Failed to load AI: {e}. If on Cloud, the model might be too large for the 1GB RAM limit.")
    st.stop()

# ---------------------------------------------------
# Helper Logic
# ---------------------------------------------------
def get_room_desc(w, l, h):
    area = w * l
    size = "small" if area < 120 else "spacious"
    ceiling = "high ceilings" if h > 10 else "standard"
    return f"{size} room with {ceiling}"

# ---------------------------------------------------
# UI Sidebar
# ---------------------------------------------------
st.sidebar.header("Room Settings")
room_type = st.sidebar.selectbox("Room", ["Living Room", "Study Room", "Bedroom", "TV Lounge"])
budget = st.sidebar.selectbox("Budget Style", ["lower", "middle", "higher"])
width = st.sidebar.slider("Width (ft)", 6, 40, 12)
length = st.sidebar.slider("Length (ft)", 6, 60, 15)
height = st.sidebar.slider("Height (ft)", 7, 18, 9)

color_theme = st.text_input("Color Theme", "modern oak wood and cream")

# ---------------------------------------------------
# Generation
# ---------------------------------------------------
if st.button("Generate Design"):
    seed = random.randint(0, 999999)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # We use fewer steps on CPU to avoid time-outs
    num_steps = 20 if device == "cpu" else 35

    with st.spinner(f"Designing your {room_type}... (Running on {device.upper()})"):
        prompt = f"""
        (Interior view of the INSIDE of a {room_type}), (looking from within the room).
        {get_room_desc(width, length, height)}. Style: {budget} budget.
        Theme: {color_theme}. Highly detailed, 8k, professional interior photography.
        """
        negative = "exterior, house facade, grass, trees, sky, roof, blurry, distorted"

        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        st.image(image, caption=f"{room_type} Design (Seed: {seed})", use_container_width=True)

        # Download logic
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button("Download PNG", buf.getvalue(), f"{room_type}.png", "image/png")
        
