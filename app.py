import streamlit as st
from openai import OpenAI
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="🏠 AI Interior Design Studio", 
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
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
    box-shadow: 0 15px 30px rgba(102,126,234,0.3);
}
.title { font-family: 'Playfair Display', serif; font-size: 3rem; margin: 0; }
.subtitle { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }
.room-card { 
    background: white; 
    border-radius: 15px; 
    padding: 1.5rem; 
    margin: 1rem 0; 
    border: 1px solid #e1e8ed;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}
.ai-image { border-radius: 15px; width: 100%; height: auto; }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI
api_ready = False
if "OPENAI_API_KEY" in st.secrets:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        api_ready = True
    except Exception as e:
        st.error(f"OpenAI Initialization Error: {e}")
else:
    st.warning("⚠️ OpenAI API Key missing! Add it to Streamlit Secrets to enable AI Generation.")

# Header
st.markdown("""<div class="main-header"><h1 class="title">🏠 AI Interior Design Studio</h1><p class="subtitle">DALL-E 3 Powered Room Planning & Visualization</p></div>""", unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎨 Design Parameters")
    rooms = ["Living Room", "Bedroom", "Drawing Room", "TV Lounge", "Kitchen", "Dining Room", "Home Office", "Gaming Room"]
    selected_room = st.selectbox("🏠 Select Room", rooms)
    
    budgets = ["Economy (Under $2K)", "Standard ($2K-$10K)", "Luxury ($10K-$50K)", "Ultra-Luxury ($50K+)"]
    selected_budget = st.selectbox("💰 Budget Level", budgets)
    
    style = st.selectbox("Design Style", ["Modern", "Minimalist", "Luxury", "Bohemian", "Industrial", "Scandinavian"])
    primary_color = st.color_picker("🎨 Theme Color", "#3498db")
    
    generate_btn = st.button("✨ Generate AI Design", type="primary", use_container_width=True)

with col2:
    if generate_btn and api_ready:
        prompt = f"Photorealistic 8K interior design of a {selected_room} in {style} style. Budget: {selected_budget}. Color theme: {primary_color}. High-end 3D render, professional lighting."
        
        with st.spinner("🎨 Creating your masterpiece..."):
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1
                )
                image_url = response.data[0].url
                st.markdown(f'<div class="room-card"><h3>✅ {selected_room} Concept</h3><img src="{image_url}" class="ai-image"></div>', unsafe_allow_html=True)
                st.success("Design Generated!")
            except Exception as e:
                st.error(f"Generation failed: {e}")
    elif generate_btn and not api_ready:
        st.error("API Key not found. Please check your secrets.toml.")
    else:
        st.info("Set your preferences on the left and click 'Generate' to see the AI concept.")

# Session State for Saved Designs
if 'designs' not in st.session_state:
    st.session_state.designs = []

st.divider()
st.subheader("💾 Saved Projects")

if st.button("💾 Save Current Configuration"):
    new_design = {"room": selected_room, "style": style, "budget": selected_budget}
    st.session_state.designs.append(new_design)
    st.toast("Design configuration saved!")

if st.session_state.designs:
    for i, d in enumerate(st.session_state.designs):
        st.write(f"{i+1}. **{d['room']}** ({d['style']}) - Budget: {d['budget']}")
else:
    st.write("No projects saved yet.")
