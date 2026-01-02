import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from io import BytesIO
import random

# ---------------------------------------------------
# App Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Interior Designer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Interior Designer")
st.write("Generate a fully designed interior room using a pretrained AI model.")

# ---------------------------------------------------
# Load Pretrained Model (Cached)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

pipe = load_model()

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def analyze_dimensions(width, length, height):
    area = width * length

    if area < 120:
        size_desc = "small compact room with space-saving furniture"
    elif area < 250:
        size_desc = "medium-sized room with balanced layout"
    else:
        size_desc = "large spacious room with luxury layout"

    if height < 9:
        ceiling = "low ceiling with recessed lighting"
    elif height < 11:
        ceiling = "standard ceiling height"
    else:
        ceiling = "high ceiling with luxury lighting"

    return f"{size_desc}, {ceiling}"

def budget_description(level):
    return {
        "lower": "affordable materials, simple furniture, minimal decor",
        "middle": "mid-range materials, modern furniture, tasteful decor",
        "higher": "premium materials, custom furniture, luxury finishes"
    }[level]

def build_prompt(room, shape, budget, color, dimension_text):
    # Added "Interior shot of the INSIDE" to force indoor view
    return f"""
(Interior shot of the INSIDE of a {room}), (looking from within the room).
Indoor photography, room shape: {shape}.
Style: {budget_description(budget)}.
Color theme: {color}.
Room proportions: {dimension_text}.
Highly detailed furniture, professional interior design, 
wide-angle lens, soft indoor lighting, 8k resolution, cinematic.
"""

# Hardened Negative Prompt to block exterior shots
NEGATIVE_PROMPT = """
exterior, outside, house facade, building exterior, street, trees, 
grass, sky, roof, garden, backyard, blurry, cartoon, anime, 
watermark, bad perspective, distorted furniture, messy, 
low resolution, grainy, fisheye lens
"""

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------
st.sidebar.header("Room Configuration")

room_name = st.sidebar.selectbox(
    "Room Type",
    ["Study Room", "Bed Room", "Living Room", "Drawing Room", "Dining Room", "TV Lounge"]
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

# ---------------------------------------------------
# Main Input
# ---------------------------------------------------
color_theme = st.text_input(
    "Color Theme",
    "modern neutral tones with wooden accents"
)

# ---------------------------------------------------
# Prompt Creation
# ---------------------------------------------------
dimension_text = analyze_dimensions(width, length, height)
prompt = build_prompt(room_name, room_shape, budget, color_theme, dimension_text)

st.subheader("Current Design Prompt")
st.info(f"The AI is instructed to stay INSIDE the {room_name}.")
st.code(prompt)

# ---------------------------------------------------
# Image Generation
# ---------------------------------------------------
if st.button("Generate Interior Design"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed to ensure different results each time
    current_seed = random.randint(0, 10**6)
    generator = torch.Generator(device=device).manual_seed(current_seed)

    with st.spinner(f"Designing your {room_name}..."):
        output = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=40, # Increased steps for better detail
            guidance_scale=8.5,     # Increased guidance to follow the prompt strictly
            generator=generator
        )

        image: Image.Image = output.images[0]

    st.image(
        image,
        caption=f"Generated {room_name} (Seed: {current_seed})",
        use_container_width=True
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    st.download_button(
        "Download Design",
        data=buffer.getvalue(),
        file_name=f"{room_name.lower()}_design.png",
        mime="image/png"
    )
