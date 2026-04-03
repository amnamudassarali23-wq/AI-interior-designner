import streamlit as st
from openai import OpenAI
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="🏠 AI Interior Design Studio", 
    page_icon="🏠",
    layout="wide"
)

# Custom Luxury CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;500&display=swap');
.main-header { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem; 
    border-radius: 20px; 
    text-align: center; 
    color: white; 
    margin-bottom: 2rem;
}
.title { font-family: 'Playfair Display', serif; font-size: 3rem; margin: 0; }
.room-card { 
    background: white; 
    border-radius: 15px; 
    padding: 1.5rem; 
    border: 1px solid #e1e8ed;
}
.ai-image { border-radius: 15px; width: 100%; height: auto; }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI Client
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("❌ API Key not found in Secrets!")

# Header
st.markdown("""<div class="main-header"><h1 class="title">🏠 AI Interior Design Studio</h1></div>""", unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎨 Plan Your Room")
    room = st.selectbox("🏠 Room Type", ["Living Room", "Bedroom", "Office", "Kitchen"])
    style = st.selectbox("✨ Style", ["Modern", "Minimalist", "Luxury", "Bohemian"])
    budget = st.selectbox("💰 Budget", ["Economy", "Luxury", "Ultra-Luxury"])
    color = st.color_picker("🎨 Theme Color", "#3498db")
    
    generate_btn = st.button("✨ Generate AI Design", type="primary", use_container_width=True)

with col2:
    if generate_btn and client:
        prompt = f"Professional 8K interior design of a {room} in {style} style, budget: {budget}, color theme: {color}. Photorealistic, 3D render."
        with st.spinner("🎨 Creating design..."):
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    n=1
                )
                image_url = response.data[0].url
                st.markdown(f'<div class="room-card"><h3>✅ {room} Concept</h3><img src="{image_url}" class="ai-image"></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
