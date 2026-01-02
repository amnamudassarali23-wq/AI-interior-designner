import streamlit as st
import torch
import random
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from PIL import Image
from io import BytesIO

# ---------------------------------------------------
# 1. App Configuration
# ---------------------------------------------------
st.set_page_config(page_title="AI Interior Designer", page_icon="🏠", layout="wide")

st.title("🏠 AI Interior Designer")
st.write("Generate interior designs. (Cloud setup takes 3-5 mins after first reboot)")

# ---------------------------------------------------
# 2. Memory-Safe Model Loader
# ---------------------------------------------------
@st.cache_resource
def load_model():
    use_cuda = torch.cuda.is_available()
    
    # FIX: Check if we are in a low-RAM cloud environment
    if use_cuda:
        # High Quality SDXL for Local GPU
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.to("cuda")
    else:
        # Light Model for Streamlit Cloud (Prevents Crashing)
        model_id = "runwayml/stable-diffusion-v1-5" 
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        pipe.to("cpu")
        # Optimization to use less RAM
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=True)
    return pipe, "cuda" if use_cuda else "cpu"

try:
    pipe, device_type = load_model()
except Exception as e:
    st.error(f"Waiting for dependencies... If you just added requirements.txt, please REBOOT the app in Streamlit Cloud. Error: {e}")
    st.stop()

# ---------------------------------------------------
# 3. Sidebar Configuration
# ---------------------------------------------------
st.sidebar.header("Room Settings")
room_name = st.sidebar.selectbox("Room Type", ["Study Room", "Living Room", "Bedroom", "Dining Room"])
budget = st.sidebar.selectbox("Budget Style", ["Lower", "Middle", "Higher"])

st.sidebar.subheader("Dimensions (ft)")
w = st.sidebar.slider("Width", 6, 40, 12)
l = st.sidebar.slider("Length", 6, 60, 15)
h = st.sidebar.slider("Height", 7, 18, 9)

color_theme = st.text_input("Color Theme", "modern warm wood and white")

# ---------------------------------------------------
# 4. Design Generation Logic
# ---------------------------------------------------
if st.button("Generate Design"):
    seed = random.randint(0, 10**6)
    generator = torch.Generator(device=device_type).manual_seed(seed)
    
    # Fewer steps on Cloud/CPU to prevent timeouts
    steps = 20 if device_type == "cpu" else 35

    with st.spinner(f"Designing your {room_name} on {device_type.upper()}..."):
        # Explicit prompt to stay INSIDE
        prompt = f"(Interior shot of the INSIDE of a {room_name}), looking from within the room. "\
                 f"Layout: {w}x{l}ft, {h}ft ceiling. Style: {budget} budget. Theme: {color_theme}. "\
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
        st.download_button("Download Image", buf.getvalue(), f"{room_name.lower()}_design.png", "image/png")
