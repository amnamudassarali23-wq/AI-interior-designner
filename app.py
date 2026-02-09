import streamlit as st
from openai import OpenAI
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# --- GAN ARCHITECTURE ---
class SimpleGenerator(nn.Module):
    """A simple GAN Generator structure"""
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 3, 64, 64)

# Function to run the GAN
def run_gan_inference():
    model = SimpleGenerator()
    # In a production app, you would use: model.load_state_dict(torch.load('model.pth'))
    noise = torch.randn(1, 100)
    with torch.no_grad():
        generated_img = model(noise).detach().cpu().squeeze(0)
    
    # Transform tensor to Image
    generated_img = (generated_img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    ndarr = generated_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Interior Designer", page_icon="🏡")

# Initialize OpenAI
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.warning("⚠️ OpenAI Key not found in Secrets. DALL-E mode will be disabled.")

st.title("🏡 Smart AI Interior Designer")

# Mode Selection
gen_mode = st.sidebar.radio("Generation Engine", ["OpenAI DALL-E 3", "Local GAN (Experimental)"])

# User Inputs
room_name = st.text_input("Room Name", placeholder="e.g. Modern Living Room")
theme_color = st.color_picker("Pick a Theme Color", "#3498db")

if st.button("Generate Design ✨"):
    if not room_name:
        st.error("Please enter a room name!")
    else:
        if gen_mode == "OpenAI DALL-E 3":
            with st.spinner("DALL-E is designing..."):
                try:
                    prompt = f"Interior design of a {room_name} with {theme_color} accents, photorealistic."
                    response = client.images.generate(model="dall-e-3", prompt=prompt, n=1)
                    st.image(response.data[0].url)
                except Exception as e:
                    st.error(f"API Error: {e}")
        
        else:
            with st.spinner("GAN is synthesizing..."):
                img = run_gan_inference()
                st.image(img, caption="GAN Generated Base Layout (Prototype Mode)", use_container_width=True)
                st.info("Note: This GAN is initialized with random weights for demonstration.")
