import streamlit as st
from openai import OpenAI
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# --- LINE 2 FIX: Robust Initialization ---
# This prevents the app from crashing if the secret isn't set up yet.
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.sidebar.warning("⚠️ OpenAI API Key not found in Secrets. DALL-E mode is disabled.")

# --- GAN ARCHITECTURE ---
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).view(-1, 3, 64, 64)

# --- APP UI ---
st.set_page_config(page_title="AI Interior Designer", page_icon="🏡")
st.title("🏡 Smart AI Interior Designer")

mode = st.radio("Choose Engine:", ["DALL-E 3 (High Res)", "GAN (Structural Concept)"])
room_name = st.text_input("Room Name:", "Modern Kitchen")

if st.button("Generate Design ✨"):
    if mode == "DALL-E 3 (High Res)":
        if client:
            with st.spinner("DALL-E is working..."):
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=f"A photorealistic interior of {room_name}",
                    n=1
                )
                st.image(response.data[0].url)
        else:
            st.error("Please add OPENAI_API_KEY to Streamlit Secrets.")
            
    else:
        with st.spinner("GAN is synthesizing..."):
            # GAN Logic
            model = SimpleGenerator()
            noise = torch.randn(1, 100)
            with torch.no_grad():
                raw_output = model(noise).squeeze(0)
            
            # MATH FIX: Convert range [-1, 1] to [0, 255]
            normalized = (raw_output + 1) / 2.0
            img_np = normalized.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            final_img = Image.fromarray(img_np).resize((512, 512))
            st.image(final_img, caption="GAN Structural Concept")
