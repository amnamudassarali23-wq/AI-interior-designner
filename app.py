import streamlit as st
from openai import OpenAI
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import numpy as np

# --- 1. ROBUST GAN ARCHITECTURE ---
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 3, 4, 2, 1, bias=False),
            nn.Tanh() # Outputs range [-1, 1]
        )

    def forward(self, x):
        return self.main(x)

def get_gan_result():
    try:
        model = SimpleGenerator()
        noise = torch.randn(1, 100, 1, 1)
        with torch.no_grad():
            output = model(noise).squeeze(0)
        
        # FIX: Normalize [-1, 1] -> [0, 255]
        output = (output + 1) / 2.0
        img_array = output.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array).resize((512, 512))
    except Exception as e:
        # FALLBACK: If GAN fails, create a procedural "Blueprint" style image
        img = Image.new('RGB', (512, 512), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((150,250), "GAN Layout Sketch", fill=(255,255,0))
        return img

# --- 2. STREAMLIT UI ---
st.set_page_config(page_title="AI Interior Designer", layout="wide")
st.title("🏡 Smart AI Interior Designer")

# Secret handling for GitHub/Streamlit Cloud
api_key = st.secrets.get("OPENAI_API_KEY", None)

tab1, tab2 = st.tabs(["Modern Designer (DALL-E)", "Structural GAN (Local)"])

with tab1:
    st.subheader("High-Detail Photorealistic Design")
    room_name = st.text_input("What room are we designing?", "Luxury Executive Office")
    if st.button("Generate with DALL-E"):
        if not api_key:
            st.error("Missing API Key! Add it to Streamlit Secrets.")
        else:
            client = OpenAI(api_key=api_key)
            with st.spinner("Analyzing aesthetics..."):
                res = client.images.generate(model="dall-e-3", prompt=f"Interior design of {room_name}")
                st.image(res.data[0].url)

with tab2:
    st.subheader("GAN Structural Blueprint")
    st.write("This generates a raw structural concept using a Local GAN.")
    if st.button("Synthesize GAN Layout"):
        with st.spinner("Calculating tensors..."):
            img = get_gan_result()
            st.image(img, caption="GAN-Generated Spatial Concept")
            st.info("💡 Note: Local GANs without pre-trained weights show abstract patterns.")
