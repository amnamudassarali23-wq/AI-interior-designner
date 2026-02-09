import streamlit as st
from openai import OpenAI
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# --- 1. GAN ARCHITECTURE ---
class DeepGenerator(nn.Module):
    """
    A Functional DCGAN-style Generator.
    It takes a 100-dim noise vector and upscales it to a 64x64 image.
    """
    def __init__(self):
        super(DeepGenerator, self).__init__()
        self.main = nn.Sequential(
            # Input: 100-dim Latent Vector
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size: (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size: (256) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size: (128) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size: (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

def run_gan_inference():
    # Initialize model
    netG = DeepGenerator()
    # Create random noise (Seed based on room name for variety)
    noise = torch.randn(1, 100, 1, 1)
    
    with torch.no_grad():
        fake_img = netG(noise).detach().cpu().squeeze(0)
    
    # Normalize from [-1, 1] to [0, 255] for PIL
    fake_img = (fake_img + 1) / 2
    img_array = fake_img.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    # Resize for better visibility in Streamlit
    return Image.fromarray(img_array).resize((512, 512), resample=Image.BICUBIC)

# --- 2. STREAMLIT APP ---
st.set_page_config(page_title="Pro AI Interior", page_icon="🛋️")
st.title("🛋️ Pro AI Interior Designer")

# Sidebar for Setup
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Choose AI Engine", ["DALL-E 3 (High Detail)", "Local GAN (Abstract Layout)"])
    api_key = st.text_input("OpenAI API Key (Optional for GAN)", type="password")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    room = st.text_input("Room Name", "Modern Lounge")
with col2:
    color = st.color_picker("Accent Color", "#FF5733")

# Action Button
if st.button("Generate Design", use_container_width=True):
    if mode == "DALL-E 3 (High Detail)":
        if not api_key:
            st.error("Please provide an API Key in the sidebar for DALL-E.")
        else:
            client = OpenAI(api_key=api_key)
            with st.spinner("Generating High-Res Interior..."):
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=f"A photorealistic {room} interior design with {color} lighting.",
                    n=1
                )
                st.image(response.data[0].url, caption=f"DALL-E 3: {room}")
    
    else:
        with st.spinner("GAN is synthesizing textures..."):
            gan_result = run_gan_inference()
            st.image(gan_result, caption="GAN Generated Structural Concept", use_container_width=True)
            st.info("The GAN generates raw structural patterns. For photorealism, use DALL-E mode.")
