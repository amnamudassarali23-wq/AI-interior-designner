import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from io import BytesIO

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Interior Designer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Interior Designer (Pretrained Model)")
st.write("Generate a fully designed interior room based on your inputs.")

# ---------------------------------------------------
# Load Pretrained Model
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
        size = "small room with space-saving furniture and minimal layout"
    elif area < 250:
        size = "medium-sized room with balanced furniture layout"
    else:
        size = "large spacious room with luxury layout and statement furniture"

    if height < 9:
        ceiling = "low ceiling with recessed lighting"
    elif height < 11:
        ceiling = "standard ceiling height"
    else:
        ceiling = "high ceiling with pendant or chandelier lighting"

    return f"{size}, {ceiling}"

def budget_description(level):
    return {
        "lower": "affordable materials, simple furniture, minimal decor",
        "middle": "mid-range materials, modern furniture, tasteful decor",
        "higher": "premium materials, custom furniture, luxury finishes"
    }[level]

def build_prompt(room, shape, budget, color, dimension_text):
    return f"""
Photorealistic interior design of a fully furnished {room}.
Room shape: {shape}.
Design style based on budget: {budget_description(budget)}.
Color theme: {color}.
Room proportions: {dimension_text}.

Highly realistic interior, proper furniture placement,
soft natural lighting, professional interior visualization,
wide-angle camera view, ultra-detailed, 4k quality.
"""

NEGATIVE_PROMPT = """
blurry, low quality, cartoon, anime, watermark,
distorted furniture, bad perspective, cluttered room
"""

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------
st.sidebar.header("🛠 Room Inputs")

room_name = st.sidebar.selectbox(
    "Room Name",
    ["Bed Room", "Study Room", "Living Room", "Drawing Room", "Dining Room", "TV Lounge"]
)

room_shape = st.sidebar.selectbox(
    "Room Shape",
    ["Square", "Rectangle", "L-Shape"]
)

budget = st.sidebar.selectbox(
    "Budget Status",
    ["lower", "middle", "higher"]
)

st.sidebar.subheader("📐 Room Dimensions (feet)")
width = st.sidebar.number_input("Width", 6.0, 40.0, 12.0)
length = st.sidebar.number_input("Length", 6.0, 60.0, 15.0)
height = st.sidebar.number_input("Height", 7.0, 18.0, 9.0)

# ---------------------------------------------------
# Main Input
# ---------------------------------------------------
color_theme = st.text_input(
    "🎨 Enter Color Theme",
    "warm beige and walnut wood"
)

# ---------------------------------------------------
# Prompt Generation
# ---------------------------------------------------
dimension_text = analyze_dimensions(width, length, height)
prompt = build_prompt(room_name, room_shape, budget, color_theme, dimension_text)

st.subheader("🧠 Design Prompt (Auto Generated)")
st.code(prompt)

# ---------------------------------------------------
# Generate Image
# ---------------------------------------------------
if st.button("🎨 Generate Designed Room"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(42)

    with st.spinner("Generating interior design..."):
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=35,
            guidance_scale=7.5,
            generator=generator
        )

        image: Image.Image = result.images[0]

    st.image(
        image,
        caption=f"{room_name} | {room_shape} | {budget}",
        use_container_width=True
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    st.download_button(
        "⬇ Download Image",
        data=buffer.getvalue(),
        file_name="interior_design.png",
        mime="image/png"
    )
